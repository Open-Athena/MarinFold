"""Make one durable, bounded regional input copy for restartable GPU work."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import s3fs

from runtime import HERE, READER, ROOT, TARGETS

class SingleAttemptS3(s3fs.S3FileSystem):
    """Do not silently repeat a cross-region copy after a transport failure."""

    retries = 1


def main() -> None:
    """Reserve bytes durably before each copy; refuse a cumulative 7 GB excess."""
    manifests = json.loads((HERE / 'regional_inputs_manifest.json').read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', choices=sorted(manifests['mirrors']), default='RNO2A')
    args = parser.parse_args()
    regional = manifests['mirrors'][args.region]
    fs = SingleAttemptS3(**({} if os.environ.get('FSSPEC_S3') else {'profile': 'cw'}),
        endpoint_url='https://cwobject.com',
        config_kwargs={'s3': {'addressing_style': 'virtual'}, 'retries': {'max_attempts': 0},
                       'connect_timeout': 10, 'read_timeout': 300})
    location = fs.call_s3('head_bucket', Bucket=regional.removeprefix('s3://').split('/')[0])
    if location['ResponseMetadata']['HTTPHeaders']['x-amz-bucket-region'] != args.region:
        raise ValueError('Mirror bucket is not colocated with the selected GPU region')
    inputs = []
    for source, subdir in [(READER, 'reader'), (ROOT + '/shared-generator-v1', 'head')]:
        for item in fs.ls(source.removeprefix('s3://'), detail=True):
            if item['type'] != 'file' or item['name'].endswith('training_state.pt'):
                continue
            inputs.append(dict(source=item['name'], relative=subdir + '/' + Path(item['name']).name,
                               bytes=item['size']))
    inputs.append(dict(source=TARGETS.removeprefix('s3://'), relative='targets.parquet',
                       bytes=fs.size(TARGETS.removeprefix('s3://'))))
    size = sum(item['bytes'] for item in inputs)
    if size >= 7_000_000_000:
        raise ValueError(f'Input mirror exceeds its 7 GB cap: {size}')
    root = regional.removeprefix('s3://')
    ledger_key = root + '/transfer-stream.json'
    if fs.exists(ledger_key):
        ledger = json.loads(fs.cat(ledger_key))
        if ledger['inputs'] != inputs:
            raise ValueError('Source inventory differs from the frozen mirror plan')
    else:
        ledger = dict(inputs=inputs, input_bytes=size, reserved_bytes=0, completed=[], objects={},
                      created_at=datetime.now(UTC).isoformat(), destination=regional)
        bucket, key = ledger_key.split('/', 1)
        fs.call_s3('put_object', Bucket=bucket, Key=key, Body=json.dumps(ledger).encode(), IfNoneMatch='*')
    for item in sorted(inputs, key=lambda item: item['bytes'], reverse=True):
        destination = root + '/' + item['relative']
        if item['relative'] in ledger['completed']:
            if fs.size(destination) != item['bytes']:
                raise ValueError('Completed mirror object changed: ' + destination)
            continue
        print('[exp368-mirror] copying ' + item['relative'], item['bytes'], flush=True)
        bucket, key = destination.split('/', 1)
        if item['relative'] not in ledger['objects']:
            upload = fs.call_s3('create_multipart_upload', Bucket=bucket, Key=key)
            ledger['objects'][item['relative']] = dict(upload_id=upload['UploadId'], parts=[])
            fs.pipe_file(ledger_key, json.dumps(ledger, indent=2).encode())
        progress = ledger['objects'][item['relative']]
        block = 32 * 1024 * 1024
        for index, start in enumerate(range(0, item['bytes'], block), 1):
            if index <= len(progress['parts']):
                continue
            end = min(item['bytes'], start + block)
            # Reserve before the cross-region GET. Retried or ambiguous chunks
            # count again, while completed multipart pieces survive preemption.
            if ledger['reserved_bytes'] + end - start > 7_000_000_000:
                raise ValueError('Retry would exceed the durable transfer reservation cap')
            ledger['reserved_bytes'] += end - start
            fs.pipe_file(ledger_key, json.dumps(ledger, indent=2).encode())
            payload = fs.cat_file(item['source'], start=start, end=end)
            if len(payload) != end - start:
                raise ValueError('Source returned an incomplete range')
            part = fs.call_s3('upload_part', Bucket=bucket, Key=key, UploadId=progress['upload_id'],
                             PartNumber=index, Body=payload)
            progress['parts'].append(dict(PartNumber=index, ETag=part['ETag']))
            fs.pipe_file(ledger_key, json.dumps(ledger, indent=2).encode())
            if index % 16 == 0:
                print('[exp368-mirror] committed bytes', end, flush=True)
        if not fs.exists(destination):
            fs.call_s3('complete_multipart_upload', Bucket=bucket, Key=key, UploadId=progress['upload_id'],
                       MultipartUpload={'Parts': progress['parts']})
        if fs.size(destination) != item['bytes']:
            raise ValueError('Mirror size mismatch: ' + destination)
        ledger['completed'].append(item['relative'])
        fs.pipe_file(ledger_key, json.dumps(ledger, indent=2).encode())
    ledger['complete_at'] = datetime.now(UTC).isoformat()
    fs.pipe_file(root + '/complete.json', json.dumps(ledger, indent=2).encode())
    (HERE / 'data').mkdir(exist_ok=True)
    (HERE / 'data/regional_mirror.json').write_text(json.dumps(ledger, indent=2) + '\n')
    print(json.dumps(dict(destination=regional, bytes=size, objects=len(inputs))))


if __name__ == '__main__':
    main()
