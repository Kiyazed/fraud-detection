def create_time_features(df):

    df["time_since_signup"] = (
        df["purchase_time"] -
        df["signup_time"]
    ).dt.total_seconds()

    df["hour_of_day"] = (
        df["purchase_time"]
        .dt.hour
    )

    df["day_of_week"] = (
        df["purchase_time"]
        .dt.dayofweek
    )

    return df
def create_transaction_frequency(df):

    transaction_count = (
        df.groupby("user_id")
        .size()
        .reset_index(name="transaction_count")
    )

    df = df.merge(
        transaction_count,
        on="user_id"
    )

    return df