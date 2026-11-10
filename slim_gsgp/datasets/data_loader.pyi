from typing import Literal, overload

import pandas as pd
from torch import Tensor

@overload
def load_pandas_df(
    df: pd.DataFrame,
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_pandas_df(
    df: pd.DataFrame,
    *,
    X_y: Literal[False] = False,
) -> Tensor: ...
@overload
def load_resid_build_sale_price(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_resid_build_sale_price(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_istanbul(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_istanbul(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_airfoil(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_airfoil(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_bike_sharing(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_bike_sharing(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_boston(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_boston(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_breast_cancer(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_breast_cancer(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_concrete_slump(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_concrete_slump(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_concrete_strength(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_concrete_strength(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_diabetes(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_diabetes(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_efficiency_heating(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_efficiency_heating(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_efficiency_cooling(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_efficiency_cooling(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_forest_fires(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_forest_fires(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_parkinson_updrs(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_parkinson_updrs(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_ld50(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_ld50(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_ppb(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_ppb(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
@overload
def load_bioav(
    *,
    X_y: Literal[True],
) -> tuple[Tensor, Tensor]: ...
@overload
def load_bioav(
    *,
    X_y: Literal[False] = False,
) -> pd.DataFrame: ...
