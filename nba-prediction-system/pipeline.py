
"""
NBA Prediction System Pipeline

This script orchestrates the full NBA prediction system pipeline, including:
1. Data fetching
2. Data processing
3. Model training
4. Prediction generation

It can run the full pipeline or individual steps based on command-line arguments.
"""

from pathlib import Path
import logging
import argparse
import os
import time
from datetime import datetime
from config.paths import *
from utils.logging import setup_logger

# Set up pipeline logger
logger = setup_logger("pipeline", None)

def run_fetch_pipeline():
    """Run data fetching pipeline to collect raw data"""
    logger.info("Starting data fetching pipeline")
    start_time = time.time()
    
    # Import modules here to avoid circular imports
    from scripts.fetch import team_roster, upcoming_games
    
    # Run data fetching scripts
    logger.info("Fetching team rosters")
    team_roster.main()
    
    logger.info("Fetching upcoming games")
    upcoming_games.main()
    
    # Add player stats fetcher once it's implemented
    # logger.info("Fetching player stats")
    # player_stats.main()
    
    duration = time.time() - start_time
    logger.info(f"Data fetching pipeline completed in {duration:.2f} seconds")

def run_process_pipeline():
    """Run data processing pipeline to clean and enhance data"""
    logger.info("Starting data processing pipeline")
    start_time = time.time()
    
    # Import modules here to avoid circular imports
    # from scripts.process import clean_stats
    
    # Run data processing scripts
    # logger.info("Cleaning stats")
    # clean_stats.main()
    
    duration = time.time() - start_time
    logger.info(f"Data processing pipeline completed in {duration:.2f} seconds")
    
def run_model_pipeline():
    """Run modeling pipeline to train models and generate predictions"""
    logger.info("Starting modeling pipeline")
    start_time = time.time()
    
    # Import modules here
    # from models import predictions
    
    # Run model training
    # logger.info("Training prediction models")
    # predictions.train_models()
    
    # Generate predictions
    # logger.info("Generating predictions")
    # predictions.generate_predictions()
    
    duration = time.time() - start_time
    logger.info(f"Modeling pipeline completed in {duration:.2f} seconds")
    
def run_full_pipeline():
    """Run complete pipeline from data fetching to predictions"""
    logger.info("Starting full NBA prediction pipeline")
    overall_start = time.time()
    
    # Run each pipeline step
    run_fetch_pipeline()
    run_process_pipeline()
    run_model_pipeline()
    
    overall_duration = time.time() - overall_start
    logger.info(f"Full pipeline completed in {overall_duration:.2f} seconds")

def main():
    """Main function to parse arguments and run selected pipeline steps"""
    # Ensure all necessary directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATA_DIR / 'raw', exist_ok=True)
    os.makedirs(DATA_DIR / 'processed', exist_ok=True)
    os.makedirs(DATA_DIR / 'features', exist_ok=True)
    os.makedirs(DATA_DIR / 'predictions', exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NBA Prediction Pipeline")
    parser.add_argument("--step", choices=["fetch", "process", "model", "all"], 
                      default="all", help="Pipeline step to run")
    args = parser.parse_args()
    
    logger.info(f"Running pipeline with step: {args.step}")
    
    # Run selected pipeline step
    if args.step == "fetch":
        run_fetch_pipeline()
    elif args.step == "process":
        run_process_pipeline()
    elif args.step == "model":
        run_model_pipeline()
    else:
        run_full_pipeline()

if __name__ == "__main__":
    main()
