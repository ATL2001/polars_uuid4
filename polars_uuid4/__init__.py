# expression_lib/__init__.py
import polars as pl
import polars.selectors as cs
from polars.utils.udfs import _get_shared_lib_location


# Boilerplate needed to inform Polars of the location of binary wheel.
lib = _get_shared_lib_location(__file__)

@pl.api.register_expr_namespace("uuid")
class Polars_UUID:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def create_uuid4(self) -> pl.Expr:
        return self._expr._register_plugin(
            lib=lib,
            symbol="create_uuid4",
            is_elementwise=True,
        )
    
@pl.api.register_dataframe_namespace("uuid")
class Polars_df_UUID:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def with_uuid4(self, col_name="uuid")->pl.DataFrame:
        str_cols = self._df.select(pl.col(pl.Utf8)).head().columns
        if len(str_cols)>0:
            col = pl.col(str_cols[0])
        else:
            col = cs.first()
        return self._df.with_columns(
            col.uuid.create_uuid4().alias(col_name)
        )
    
@pl.api.register_lazyframe_namespace("uuid")
class Polars_lf_UUID:
    def __init__(self, lf: pl.LazyFrame):
        self._lf = lf

    def with_uuid4(self, col_name="uuid")->pl.LazyFrame:
        str_cols = self._lf.select(pl.col(pl.Utf8)).head().columns
        if len(str_cols)>0:
            col = pl.col(str_cols[0])
        else:
            col = cs.first()
        return self._lf.with_columns(col.uuid.create_uuid4().alias(col_name))