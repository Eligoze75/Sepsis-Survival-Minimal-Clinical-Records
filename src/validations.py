import pandera.pandas as pa


TARGET_POSSIBLE_VALUES = [0, 1]
SEX_POSSIBLE_VALUES = ["male", "female"]
AGE_DTYPE = int
SEX_DTYPE = str
EPISODE_DTYPE = int
TARGET_DTYPE = int
AGE_RANGE = pa.Check.between(0, 130)
EPISODE_RANGE = pa.Check.between(0, 15)
FEATURES = ["age", "sex", "episode_number"]


def check_target_ratio(x):
    return (x.mean() > 0.5) & (x.mean() < 0.95)


def check_empty_rows(df):
    return ~(df[FEATURES].isna().all(axis=1)).any()


# This schema only validates for the presence of all required variables
# and ensures that the possible values in the features make sense
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
        pa.Check(check_empty_rows, error="Empty rows found."),
    ],
)

# This schema validates for the presence of all required variables
# ensures that the possible values in the features make sense with context from the EDA
test_schema = pa.DataFrameSchema(
    {
        "hospital_outcome": pa.Column(
            TARGET_DTYPE,
            checks=[
                pa.Check.isin(TARGET_POSSIBLE_VALUES),
                pa.Check(check_target_ratio),
            ],
            nullable=False,
        ),
        "age": pa.Column(AGE_DTYPE, AGE_RANGE, nullable=False),
        "sex": pa.Column(SEX_DTYPE, pa.Check.isin(SEX_POSSIBLE_VALUES), nullable=False),
        "episode_number": pa.Column(EPISODE_DTYPE, EPISODE_RANGE, nullable=False),
    },
    checks=[
        pa.Check(check_empty_rows, error="Empty rows found."),
    ],
)

# This schema is meant to be use to validate data for prediction
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
        pa.Check(check_empty_rows, error="Empty rows found."),
    ],
)
