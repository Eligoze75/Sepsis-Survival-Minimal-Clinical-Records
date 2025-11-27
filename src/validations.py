import pandera.pandas as pa


TARGET_POSSIBLE_VALUES = [0, 1]
SEX_POSSIBLE_VALUES = ["male", "female"]
AGE_DTYPE = int
SEX_DTYPE = str
EPISODE_DTYPE = int
TARGET_DTYPE = int
AGE_RANGE = pa.Check.between(0, 130)
EPISODE_RANGE = pa.Check.between(0, 15)


initial_schema = pa.DataFrameSchema(
    {
        "hospital_outcome": pa.Column(
            TARGET_DTYPE, pa.Check.isin(TARGET_POSSIBLE_VALUES), nullable=False
        ),
        "age": pa.Column(AGE_DTYPE, AGE_RANGE, nullable=True),
        "sex": pa.Column(SEX_DTYPE, pa.Check.isin(SEX_POSSIBLE_VALUES), nullable=True),
        "episode_number": pa.Column(EPISODE_DTYPE, EPISODE_RANGE, nullable=True),
    },
    checks=[
        pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found."),
    ],
)

test_schema = pa.DataFrameSchema(
    {
        "hospital_outcome": pa.Column(
            TARGET_DTYPE, pa.Check.isin(TARGET_POSSIBLE_VALUES), nullable=False
        ),
        "age": pa.Column(AGE_DTYPE, AGE_RANGE, nullable=False),
        "sex": pa.Column(SEX_DTYPE, pa.Check.isin(SEX_POSSIBLE_VALUES), nullable=False),
        "episode_number": pa.Column(EPISODE_DTYPE, EPISODE_RANGE, nullable=False),
    },
    checks=[
        pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found."),
    ],
)

prediction_schema = pa.DataFrameSchema(
    {
        "hospital_outcome": pa.Column(
            TARGET_DTYPE, pa.Check.isin(TARGET_POSSIBLE_VALUES), required=False
        ),
        "age": pa.Column(AGE_DTYPE, AGE_RANGE, nullable=False),
        "sex": pa.Column(SEX_DTYPE, pa.Check.isin(SEX_POSSIBLE_VALUES), nullable=False),
        "episode_number": pa.Column(EPISODE_DTYPE, EPISODE_RANGE, nullable=False),
    },
    checks=[
        pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found."),
    ],
)
