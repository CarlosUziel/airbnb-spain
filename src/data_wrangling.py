import datetime
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd


def airbnb_avg_price(listings_path: Path, n_hoods: int = 1):
    """
    Provide the data to answer the following question:
        "What is the average price of each location type per neighbourhood? What are the
        most expensive neighbourhoods on average?"

    Args:
        listings_path: File path to a dataframe where each row is a description of each
            Airbnb listing.
        n_hoods: Number of top neighbourhoods to include.

    """
    logging.info(f"Processing {listings_path}...")

    # 0. Setup
    listings_df = pd.read_csv(listings_path, index_col=0, low_memory=False).dropna(
        subset=["price", "room_type", "neighbourhood_cleansed"]
    )

    # remove non-string values from relevant columns
    listings_df = listings_df[
        pd.to_numeric(listings_df["room_type"], errors="coerce").isna()
        & pd.to_numeric(listings_df["price"], errors="coerce").isna()
        & pd.to_numeric(listings_df["neighbourhood_cleansed"], errors="coerce").isna()
    ]

    price_str_to_float = (
        lambda x: float(x.replace("$", "").replace(",", ""))
        if isinstance(x, str)
        else x
    )
    listings_df["price_num"] = listings_df["price"].apply(price_str_to_float)

    # 1. Get average price per neighbourhood and listing type
    listings_df = listings_df.dropna(subset=["neighbourhood_cleansed"])
    df = (
        listings_df[["neighbourhood_cleansed", "room_type", "price_num"]]
        .groupby(["neighbourhood_cleansed", "room_type"])
        .mean(numeric_only=True)
        .round(2)
    )

    # 2. Sort by summing up all room types
    sorted_sums = df["price_num"].groupby(level=0).sum().sort_values(ascending=False)
    df = df.reindex(sorted_sums.index, level=0).unstack(level=1)["price_num"]

    # 3. Select most expensive neighborhood for each room type
    most_expensive_hoods = {
        room_type: " | ".join(
            [
                s.title()
                for s in df[room_type].sort_values(ascending=False).iloc[:n_hoods].index
            ]
        )
        for room_type in df.columns
    }
    logging.info(f"Finished processing {listings_path}.")

    return df, most_expensive_hoods


def airbnb_avg_accept_rate(listings_path: Path, n_hoods: int = 1):
    """
    Provide the data to answer the following question:
        "What is the average host acceptance rate per location type and neighborhood? In
        which neighbourhoods is it the highest and in which the lowest?"

    Args:
        listings_path: File path to a dataframe where each row is a description of each
            Airbnb listing.
        n_hoods: Number of top neighbourhoods to include.

    """
    logging.info(f"Processing {listings_path}...")

    # 0. Setup
    listings_df = pd.read_csv(listings_path, index_col=0, low_memory=False).dropna(
        subset=["host_acceptance_rate", "room_type", "neighbourhood_cleansed"]
    )

    # remove non-string values from relevant columns
    listings_df = listings_df[
        pd.to_numeric(listings_df["room_type"], errors="coerce").isna()
        & pd.to_numeric(listings_df["host_acceptance_rate"], errors="coerce").isna()
        & pd.to_numeric(listings_df["neighbourhood_cleansed"], errors="coerce").isna()
    ]

    perc_str_to_float = (
        lambda x: float(x.replace("%", "")) / 100 if isinstance(x, str) else x
    )
    listings_df["host_acceptance_rate_num"] = listings_df["host_acceptance_rate"].apply(
        perc_str_to_float
    )

    # 1. Get average acceptance rate per neighbourhood and listing type
    listings_df = listings_df.dropna(subset=["neighbourhood_cleansed"])
    df = (
        listings_df[["neighbourhood_cleansed", "room_type", "host_acceptance_rate_num"]]
        .groupby(["neighbourhood_cleansed", "room_type"])
        .mean(numeric_only=True)
        .round(2)
    )

    # 2. Sort by summing up all room types
    sorted_sums = (
        df["host_acceptance_rate_num"]
        .groupby(level=0)
        .sum()
        .sort_values(ascending=False)
    )
    df = df.reindex(sorted_sums.index, level=0).unstack(level=1)[
        "host_acceptance_rate_num"
    ]

    # 3. Select the neighborhood with the highest acceptance rate for each room type
    highest_accept_rate_hoods = {
        room_type: " | ".join(
            [
                s.title()
                for s in df[room_type].sort_values(ascending=False).iloc[:n_hoods].index
            ]
        )
        for room_type in df.columns
    }
    logging.info(f"Finished processing {listings_path}.")

    return df, highest_accept_rate_hoods


def airbnb_hood_hosts(listings_path: Path, n_hoods: int = 1):
    """
    Provide the data to answer the following question:
        "How is competition in each neighbourhood? What number and proportion of
        listings belong to hosts owning different numbers of locations?"

    Args:
        listings_path: File path to a dataframe where each row is a description of each
            Airbnb listing.
        n_hoods: Number of top neighbourhoods to include.

    """
    logging.info(f"Processing {listings_path}...")

    # 0. Setup
    listings_df = pd.read_csv(listings_path, low_memory=False).dropna(
        subset=["id", "host_id", "neighbourhood_cleansed"]
    )

    # 1. Get number of listings per host in each neighbourhood
    df = (
        listings_df[["host_id", "neighbourhood_cleansed", "id"]]
        .groupby(["host_id", "neighbourhood_cleansed"])
        .count()["id"]
        .sort_values(ascending=False)
        .unstack(level=0)
        .fillna(0)
    )

    # 2. Bin hosts into groups depending on how many listings they own
    neighborhood_hosts_groups = defaultdict(dict)
    for neighborhood, neighborhood_hosts in df.iterrows():
        # Hosts with only one listing
        total = sum(neighborhood_hosts == 1)
        p = total / sum(neighborhood_hosts >= 1) * 100
        neighborhood_hosts_groups[neighborhood]["1"] = f"{total} ({p:.2f}%)"

        # Hosts managing between 2 and 5 listings
        total = sum((neighborhood_hosts >= 2) & (neighborhood_hosts <= 5))
        p = total / sum(neighborhood_hosts >= 1) * 100
        neighborhood_hosts_groups[neighborhood]["2_to_5"] = f"{total} ({p:.2f}%)"

        # Hosts managing between 6 to 20 listings
        total = sum((neighborhood_hosts >= 6) & (neighborhood_hosts <= 20))
        p = total / sum(neighborhood_hosts >= 1) * 100
        neighborhood_hosts_groups[neighborhood]["6_to_20"] = f"{total} ({p:.2f}%)"

        # Hosts managing 21 or more listings
        total = sum(neighborhood_hosts >= 21)
        p = total / sum(neighborhood_hosts >= 1) * 100
        neighborhood_hosts_groups[neighborhood]["21_to_many"] = f"{total} ({p:.2f}%)"

    # 3. Final host group count per neighbourhood
    host_counts_df = pd.DataFrame(neighborhood_hosts_groups).transpose()
    host_counts_df.sort_values(
        by=host_counts_df.columns.tolist(),
        key=lambda x: [int(r.split(" ")[0]) for r in x],
        ascending=False,
    )

    # 4. Select neighbourhoods with highest concentration
    most_dense_hoods = {
        host_group: " | ".join(
            [
                s.title()
                for s in host_counts_df[host_group]
                .sort_values(ascending=False)
                .iloc[:n_hoods]
                .index
            ]
        )
        for host_group in host_counts_df.columns
    }
    logging.info(f"Finished processing {listings_path}.")

    return host_counts_df, most_dense_hoods


def airbnb_avg_profit(
    listings_path: Path, calendar_path: Path, n_weeks: int = 8, n_hoods: int = 1
):
    """
    Provide the data to answer the following question:
        "What is the expected average profit per room type and neighborhood when looking
        at the reservations for the next weeks? What is the neighbourhood expected to be
        the most profitable in that period?"

    Args:
        listings_path: File path to a dataframe where each row is a description of each
            Airbnb listing.
        calendar_path: File path to a dataframe where each row is a calendar entry for
            one of the Airbnb listings.
        n_weeks: How many weeks into the future to look at.
        n_hoods: Number of top neighbourhoods to include.

    """
    logging.info(f"Processing {listings_path} and {calendar_path}...")

    # 0. Setup
    # 0.1. Load and transform calendar data
    calendar_df = pd.read_csv(calendar_path, low_memory=False)

    calendar_df["date"] = pd.to_datetime(calendar_df["date"])
    calendar_df = calendar_df[
        calendar_df["date"]
        <= (calendar_df["date"].min() + datetime.timedelta(weeks=n_weeks))
    ]

    price_str_to_float = (
        lambda x: float(x.replace("$", "").replace(",", ""))
        if isinstance(x, str)
        else x
    )
    # remove non-string/non-numeric values from relevant columns
    calendar_df = calendar_df[
        ~pd.to_numeric(calendar_df["listing_id"], errors="coerce").isna()
        & pd.to_numeric(calendar_df["adjusted_price"], errors="coerce").isna()
    ]
    calendar_df["listing_id"] = pd.to_numeric(
        calendar_df["listing_id"], errors="coerce"
    )
    calendar_df["adjusted_price_num"] = calendar_df["adjusted_price"].apply(
        price_str_to_float
    )

    # 0.2. Load and transform listings data
    listings_df = pd.read_csv(listings_path, low_memory=False).dropna(
        subset=["room_type", "neighbourhood_cleansed", "id"]
    )

    # remove non-string/non-numeric values from relevant columns
    listings_df = listings_df[
        pd.to_numeric(listings_df["room_type"], errors="coerce").isna()
        & pd.to_numeric(listings_df["neighbourhood_cleansed"], errors="coerce").isna()
        & ~pd.to_numeric(listings_df["id"], errors="coerce").isna()
    ]
    listings_df["id"] = pd.to_numeric(listings_df["id"], errors="coerce")

    # 0.3. Sum reservations price for each listing
    calendar_df[calendar_df["available"] == "f"][
        ["listing_id", "adjusted_price_num"]
    ].groupby("listing_id").sum(numeric_only=True).join(
        listings_df.set_index("id")
    ).to_csv(
        "tmp.csv"
    )

    # 1. Get profits for each listing
    listings_profits_df = (
        calendar_df[calendar_df["available"] == "f"][
            ["listing_id", "adjusted_price_num"]
        ]
        .groupby("listing_id")
        .sum(numeric_only=True)
        .join(listings_df.set_index("id"))
        .rename(columns={"adjusted_price_num": "total_profit"})
    )

    # 2. Average profits per neighbourhood and room type
    df = (
        listings_profits_df[["neighbourhood_cleansed", "room_type", "total_profit"]]
        .groupby(["neighbourhood_cleansed", "room_type"])
        .mean()
        .round(2)
    )

    # 3. Sort by total profit
    sorted_sums = df["total_profit"].groupby(level=0).sum().sort_values(ascending=False)
    df = df.reindex(sorted_sums.index, level=0).unstack(level=1)["total_profit"]

    # 4. Select most profitable neighborhoods for each room type
    most_profitable_hoods = {
        room_type: " | ".join(
            [
                s.title()
                for s in df[room_type].sort_values(ascending=False).iloc[:n_hoods].index
            ]
        )
        for room_type in df.columns
    }
    logging.info(f"Finished processing {listings_path} and {calendar_path}.")

    return df, most_profitable_hoods
