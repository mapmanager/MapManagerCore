import geopandas as gp
import pandas as pd


def updateDataFrame(df: gp.GeoDataFrame, ids: pd.Index, value: pd.Series):
    """
    Update a GeoDataFrame with new values based on given ids and values.

    Args:
        df (gp.GeoDataFrame): The GeoDataFrame to be updated.
        ids (pd.Index): The ids to be updated.
        value (pd.Series): The new values to be assigned to all the provided ids.

    Returns:
        list: The updated list of ids normalized to a list.
    """

    if isinstance(ids, pd.Index):
        ids = ids.tolist()

    indexes = value.index.intersection(df.index.names)
    indexLess = value.drop(indexes)
    for (i, id) in enumerate(ids):
        oldId = id

        # Update indexes when ids change
        if isinstance(id, tuple):
            id = list(id)
            id_changed = False
            for idx, name in enumerate(df.index.names):
                if name in value.index:
                    if id[idx] != value[name]:
                        id[idx] = value[name]
                        id_changed = True
            id = tuple(id)

            if id_changed:
                ids[i] = id
                df.loc[id, :] = df.loc[oldId, :]
                df.drop(oldId, inplace=True)

        else:
            name = df.index.names[0]
            if name in value.index:
                ids[i] = id = value[name]
                df.rename(index={oldId: id}, inplace=True)

        df.loc[id, indexLess.index.values] = indexLess.values

    return ids
