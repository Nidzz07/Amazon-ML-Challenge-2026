import polars as pl
import numpy as np

def build_training_set(
    candidates_path: str,
    ground_truth_parquet_path: str,
    output_train_path: str,
    output_val_path: str,
    neg_to_pos_ratio: float = 2.0,
    entity_subsample_fraction: float = 0.4,
    val_split_fraction: float = 0.2,
    seed: int = 42
):
    """
    Constructs the training and validation splits.
    - Keeps all positives.
    - Samples hard negatives (top-ranked by prior_score) at the specified ratio.
    - Subsamples entities to the given fraction.
    - Splits into train/val by ENTITY, not pair.
    """
    # Set seed for determinism
    np.random.seed(seed)
    
    # 1. Load candidates and ground truth (assuming GT is processed into positive pairs)
    # Ground truth schema expected: source1_entity_id, candidate_entity_id
    print("Loading data...")
    candidates = pl.read_parquet(candidates_path)
    positives_gt = pl.read_parquet(ground_truth_parquet_path)
    
    # Create the label column by left joining with ground truth
    positives_gt = positives_gt.with_columns(pl.lit(1, dtype=pl.UInt8).alias("label"))
    candidates = candidates.join(
        positives_gt, 
        on=["source1_entity_id", "candidate_entity_id"], 
        how="left"
    ).with_columns(pl.col("label").fill_null(0))

    # 2. Subsample entities
    print(f"Subsampling {entity_subsample_fraction * 100}% of entities...")
    unique_entities = candidates.select("source1_entity_id").unique()
    n_entities = len(unique_entities)
    n_subsampled = int(n_entities * entity_subsample_fraction)
    
    # Randomly select entities
    sampled_entities_df = unique_entities.sample(n=n_subsampled, seed=seed)
    
    # Filter candidates to only these entities
    candidates = candidates.join(sampled_entities_df, on="source1_entity_id", how="inner")

    # 3. Separate positives and negatives
    positives = candidates.filter(pl.col("label") == 1)
    negatives = candidates.filter(pl.col("label") == 0)

    print(f"Positives kept: {len(positives)}")
    
    # 4. Sample hard negatives from top-ranked non-matches
    # Since we want top-ranked negatives, we sort by 'prior_score' descending
    print("Sampling hard negatives...")
    negatives = negatives.sort(by=["source1_entity_id", "prior_score"], descending=[False, True])
    
    # Target a global ratio of negatives to positives (e.g. 2:1)
    target_negatives_count = int(len(positives) * neg_to_pos_ratio)
    
    if target_negatives_count < len(negatives):
        # Taking the head keeps the highest prior_score negatives globally
        negatives = negatives.head(target_negatives_count)
        
    print(f"Negatives kept: {len(negatives)}")

    # Combine back together
    dataset = pl.concat([positives, negatives])
    
    # 5. Split into train and val by ENTITY
    print(f"Splitting into train/val by entity (val_fraction={val_split_fraction})...")
    final_entities = dataset.select("source1_entity_id").unique()
    
    n_val = int(len(final_entities) * val_split_fraction)
    val_entities = final_entities.sample(n=n_val, seed=seed)
    
    # Flag validation entities
    val_entities = val_entities.with_columns(pl.lit(True).alias("is_val"))
    dataset = dataset.join(val_entities, on="source1_entity_id", how="left").with_columns(
        pl.col("is_val").fill_null(False)
    )
    
    train_set = dataset.filter(pl.col("is_val") == False).drop("is_val")
    val_set = dataset.filter(pl.col("is_val") == True).drop("is_val")
    
    print(f"Final Train Size: {len(train_set)} rows")
    print(f"Final Val Size: {len(val_set)} rows")
    
    # 6. Save to disk
    # Krrish's featuriser will read these to append the ~60 features
    print("Saving to parquet...")
    train_set.write_parquet(output_train_path)
    val_set.write_parquet(output_val_path)
    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates parquet from blocking")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to parsed ground truth positive pairs parquet")
    parser.add_argument("--out-train", type=str, required=True, help="Output path for train split")
    parser.add_argument("--out-val", type=str, required=True, help="Output path for validation split")
    args = parser.parse_args()
    
    build_training_set(args.candidates, args.ground_truth, args.out_train, args.out_val)
