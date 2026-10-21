import pandas as pd
from util import create_feature_store, load_entity_df, load_historical_data

pd.set_option("display.max_columns", None)

store = create_feature_store()
entity_df = load_entity_df()

_, _, test_df = load_historical_data(store=store, entity_df=entity_df)
