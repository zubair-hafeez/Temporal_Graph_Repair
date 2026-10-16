from revision.cohorts import assign_partitions


def test_partition_excludes_entire_shared_child_and_original_root():
    roots = {'old', 'crossing', 'shared_a', 'shared_b'} | {f'r{i}' for i in range(12)}
    children = {'old': {'old_child'}, 'crossing': {'old_child'},
                'shared_a': {'shared'}, 'shared_b': {'shared'}}
    children.update({f'r{i}': {f'c{i}'} for i in range(12)})
    plan = {'ranking_namespace': 'test', 'partition_sizes': {'development': 3, 'confirmation': 5}}
    parts, audit = assign_partitions(roots, children, {'old'}, {'old_child'}, 'test', plan)
    assert len(parts['development']) == 3 and len(parts['confirmation']) == 5
    assert not parts['development'] & parts['confirmation']
    assert not set.union(*parts.values()) & {'old', 'crossing', 'shared_a', 'shared_b'}
    assert audit['root_and_child_disjointness_verified']
    assert parts == assign_partitions(set(reversed(sorted(roots))), children,
                                     {'old'}, {'old_child'}, 'test', plan)[0]


def test_extending_final_block_preserves_earlier_partitions_and_old_native_roots():
    roots = {f'r{i}' for i in range(30)}
    children = {r: {f'child_{r}'} for r in roots}
    plan = {'ranking_namespace': 'test', 'partition_sizes': {
        'development': 3, 'annotation': 4, 'confirmation': 5, 'native_comparison': 2}}
    original, _ = assign_partitions(roots, children, set(), set(), 'test', plan)
    plan['partition_sizes']['native_comparison'] = 10
    extended, _ = assign_partitions(roots, children, set(), set(), 'test', plan)
    for part in ['development', 'annotation', 'confirmation']:
        assert original[part] == extended[part]
    assert original['native_comparison'] < extended['native_comparison']
