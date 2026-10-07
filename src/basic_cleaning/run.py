#!/usr/bin/env python
"""Download a W&B artifact, perform basic cleaning, and log the result."""

import argparse
import logging

import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact.
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Downloading artifact %s", args.input_artifact)
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)
    logger.info("Loaded raw data with %s rows and %s columns", *df.shape)

    df = df.drop_duplicates().reset_index(drop=True)
    df = df.dropna(subset=["price"])

    logger.info("Filtering prices between %s and %s", args.min_price, args.max_price)
    idx = df["price"].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Filter rows outside NYC geographic boundaries
    logger.info("Filtering rows outside NYC geographic boundaries")

    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    logger.info("Cleaned data has %s rows and %s columns", *df.shape)

    df.to_csv("clean_sample.csv", index=False)

    logger.info("Uploading %s to Weights & Biases", args.output_artifact)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)
    artifact.wait()

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact as given (sample csv file)",
        required=True,
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output file name",
        required=True,
    )
    parser.add_argument(
        "--output_type",
        type=str,
        help="Output artifact type",
        required=True,
    )
    parser.add_argument(
        "--output_description",
        type=str,
        help="Output artifact description",
        required=True,
    )
    parser.add_argument(
        "--min_price",
        type=float,
        help="Min price to keep",
        required=True,
    )
    parser.add_argument(
        "--max_price",
        type=float,
        help="Max price to keep",
        required=True,
    )

    args = parser.parse_args()
    go(args)
