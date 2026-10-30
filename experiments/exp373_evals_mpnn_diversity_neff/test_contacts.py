"""Check parsing invariances and retention of malformed/empty samples."""

from contacts import parse_contacts, sample_seed


def test_ring_orientation_order_and_duplicates() -> None:
    a = parse_contacts('<contact> <p1998> <p8> <contact> <p10> <p1999>',1998,30)
    b = parse_contacts('<contact> <p11> <p22> <contact> <p20> <p10> <contact> <p10> <p20>',10,30)
    assert a['contacts']==b['contacts']==[(0,10),(1,12)]
    assert b['duplicate_contacts']==1


def test_retraction_is_an_ordered_edit() -> None:
    record = parse_contacts('<contact> <p1> <p11> <retract> <p11> <p1> <contact> <p2> <p12>',0,20)
    assert record['contacts']==[(2,12)]
    assert record['retractions']==1


def test_invalid_and_truncated_statements_are_counted() -> None:
    record = parse_contacts('<contact> <p0> <p99> <contact> <p0> <p1> <contact> <p0>',0,20)
    assert record['contacts']==[]
    assert record['invalid_positions']==1
    assert record['near_diagonal']==1
    assert record['malformed_statements']==1
    assert record['unexpected_text']


def test_end_discards_following_edits_and_empty_survives() -> None:
    assert parse_contacts('<end> <contact> <p0> <p10>',0,20)['contacts']==[]
    assert parse_contacts('',0,20)['malformed_statements']==0


def test_sampling_streams_do_not_depend_on_sharding() -> None:
    seeds = {sample_seed('protein','fixed',b,k) for b in range(3) for k in range(100)}
    assert len(seeds)==300
    assert seeds.isdisjoint({sample_seed('protein','randomized',0,k) for k in range(100)})
