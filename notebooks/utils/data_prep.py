def clean_id_cols(df, id_cols):
    df[id_cols] = df[id_cols].fillna(0).astype(int).astype(str)
    df[id_cols]  = df[id_cols].apply(lambda x: x.str.strip())

    return df