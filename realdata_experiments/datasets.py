"""Dataset configurations for the real-data experiments."""

HOUSE_ALL = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long',
    'sqft_living15', 'sqft_lot15',
]

# sqft_living = sqft_above + sqft_basement, so sqft_above is dropped.
HOUSE_10D_DROP = ['id', 'date', 'sqft_above']

HOUSE_10D_TARGETS = [
    'price', 'lat', 'long', 'sqft_living', 'sqft_living15',
    'sqft_lot', 'sqft_lot15', 'grade', 'bathrooms', 'bedrooms',
]


HOUSE_LADDER_FEATURES = [c for c in HOUSE_ALL
                         if c not in HOUSE_10D_TARGETS and c not in HOUSE_10D_DROP]

DATASETS = {
    'bio_2d': dict(csv='../data/feldman/bio.csv',
                   targets=['RMSD', 'F7']),
    'bio_3d': dict(csv='../data/feldman/bio.csv',
                   targets=['RMSD', 'F7', 'F8']),
    'bio_4d': dict(csv='../data/feldman/bio.csv',
                   targets=['RMSD', 'F7', 'F8', 'F9']),

    'house_2d': dict(csv='../data/feldman/house.csv',
                     drop=['id', 'date'],
                     targets=['lat', 'price']),
    'house_3d': dict(csv='../data/feldman/house.csv',
                     drop=['id', 'date'],
                     targets=['lat', 'price', 'sqft_living15']),
    'house_4d': dict(csv='../data/feldman/house.csv',
                     drop=['id', 'date'],
                     targets=['lat', 'price', 'sqft_living15', 'sqft_lot15']),

    'blog_2d': dict(csv='../data/feldman/blog_data.csv', header=None,
                    targets=[60, 280]),
    'blog_3d': dict(csv='../data/feldman/blog_data.csv', header=None,
                    targets=[60, 61, 280]),
    'blog_4d': dict(csv='../data/feldman/blog_data.csv', header=None,
                    targets=[60, 61, 279, 280]),

    'house_10d': dict(csv='../data/feldman/house.csv',
                      drop=HOUSE_10D_DROP,
                      targets=HOUSE_10D_TARGETS),
    'house_6d':  dict(csv='../data/feldman/house.csv',
                      drop=HOUSE_10D_DROP,
                      targets=HOUSE_10D_TARGETS[:6],
                      features=HOUSE_LADDER_FEATURES),
    'house_8d':  dict(csv='../data/feldman/house.csv',
                      drop=HOUSE_10D_DROP,
                      targets=HOUSE_10D_TARGETS[:8],
                      features=HOUSE_LADDER_FEATURES),
}


ALL_RUNS = list(DATASETS)


def get(name):
    if name not in DATASETS:
        raise KeyError(f"unknown run {name!r}. Available: {sorted(DATASETS)}")
    cfg = dict(DATASETS[name])
    cfg.setdefault('header', 'infer')
    cfg.setdefault('drop', [])
    cfg.setdefault('features', None)   # None = every other column
    cfg['name'] = name
    cfg['d'] = len(cfg['targets'])
    return cfg


def _validate():
    """Sanity checks on the House configurations."""
    feats = HOUSE_LADDER_FEATURES

    overlap = set(feats) & set(HOUSE_10D_TARGETS)
    assert not overlap, f"house_10d: column is both target and feature: {overlap}"

    trio = {'sqft_living', 'sqft_above', 'sqft_basement'}
    surviving = trio - set(HOUSE_10D_DROP)
    assert len(surviving) <= 2, (
        f"house_10d: all three of {sorted(trio)} survive -- "
        f"sqft_living = sqft_above + sqft_basement would make the residual "
        f"covariance singular. Drop one."
    )

    assert len(HOUSE_10D_TARGETS) == 10, "house_10d must have exactly 10 targets"
    assert len(set(HOUSE_10D_TARGETS)) == 10, "duplicate target in house_10d"

    assert len(HOUSE_LADDER_FEATURES) == 8, "house ladder must have 8 features"


_validate()


if __name__ == '__main__':
    print(f"{'run':<12s} {'d':>3s} {'p':>6s}  targets")
    print('-' * 100)
    for name in ALL_RUNS:
        cfg = get(name)
        p = (len(HOUSE_LADDER_FEATURES) if name.startswith('house_') and cfg['d'] > 4
             else '(rest)')
        print(f"{name:<12s} {cfg['d']:>3d} {str(p):>6s}  {cfg['targets']}")
