"""Ensure cleanup cannot discard the best model or recent recovery states."""

from train import prune_checkpoints


def test_retention_preserves_best_recent_and_unrelated_files(tmp_path):
    for step in [1,2,3,4,5,6]:
        path=tmp_path/f'step-{step}'
        path.mkdir()
        (path/'model').write_text('checkpoint')
    (tmp_path/'notes').write_text('retain')
    removed=prune_checkpoints(str(tmp_path),3,str(tmp_path/'step-1'))
    assert set(removed)=={str(tmp_path/'step-2'),str(tmp_path/'step-3')}
    assert {p.name for p in tmp_path.iterdir()}=={'step-1','step-4','step-5','step-6','notes'}
