import pandas as pd


def shorten_name(name):
    if not name or not isinstance(name, str):
        return name
    parts = name.split('/')
    return parts[-1] if len(parts) > 1 else name


def apply_filters(df, filter_args):
    if not filter_args:
        return df
    
    for filter_expr in filter_args:
        if '=' not in filter_expr:
            print(f"Invalid filter format: {filter_expr}. Expected field=value")
            return None
        
        field, value = filter_expr.split('=', 1)
        
        if field not in df.columns:
            print(f"Unknown field: {field}. Available fields: {', '.join(df.columns)}")
            return None
        
        if pd.api.types.is_numeric_dtype(df[field]):
            value = pd.to_numeric(value)
        df = df[df[field] == value]
    
    if df.empty:
        print("No matching records found")
        return None
    
    return df


def get_varying_cols(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    
    varying_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        try:
            if df[col].nunique(dropna=False) > 1:
                varying_cols.append(col)
        except TypeError:
            pass
    return varying_cols

