#!/usr/bin/env python3
"""
NBA Predictions Automated Pipeline

This script handles the complete end-to-end pipeline for the NBA prediction system:
1. Data collection (player stats, projections, etc.)
2. Data processing and feature engineering
3. Model training and evaluation
4. Generating predictions and dashboards
5. Historical tracking of model performance

The system is designed to run on a schedule and keep all predictions up-to-date.
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import argparse
import logging
import subprocess
import time
import json
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import schedule
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
os.makedirs('/Users/lukesmac/Models/nba/logs', exist_ok=True)
logging.basicConfig(
    filename='/Users/lukesmac/Models/nba/logs/unified_stats_fetcher.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add handler to output to console as well
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

# Define paths
BASE_DIR = Path('/Users/lukesmac/Models')
NBA_DIR = BASE_DIR / 'nba'
PRIZEPICKS_DIR = BASE_DIR / 'prizepicks'
DATA_DIR = NBA_DIR / 'data'
MODELS_DIR = NBA_DIR / 'models'
SCRIPTS_DIR = NBA_DIR / 'scripts'
LOGS_DIR = NBA_DIR / 'logs'
DASHBOARD_DIR = NBA_DIR / 'dashboard_data'

# Create necessary directories
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
DASHBOARD_DIR.mkdir(exist_ok=True)

# Database paths
PLAYER_GAME_STATS_DB = DATA_DIR / 'player_game_stats.db'
CLEAN_STATS_DB = DATA_DIR / 'clean_stats.db'
ENHANCED_STATS_DB = DATA_DIR / 'enhanced_stats.db'
TRAIN_DB = DATA_DIR / 'clean_stats_train.db'
TEST_DB = DATA_DIR / 'clean_stats_test.db'
PROJECTIONS_DB = PRIZEPICKS_DIR / 'data/nba_projections.db'
HISTORICAL_DB = DATA_DIR / 'historical_predictions.db'

# Lock for database operations
db_lock = threading.Lock()

class PipelineTask:
    """Base class for pipeline tasks with retry logic and status tracking."""
    
    def __init__(self, name, max_retries=3, retry_delay=60):
        self.name = name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_run_time = None
        self.last_run_status = None
        self.last_run_message = None
        self.last_run_duration = None
    
    def run(self):
        """Run the task with retry logic."""
        start_time = time.time()
        for attempt in range(1, self.max_retries + 1):
            try:
                logging.info(f"Starting task: {self.name} (attempt {attempt}/{self.max_retries})")
                result = self.execute()
                end_time = time.time()
                duration = end_time - start_time
                
                self.last_run_time = datetime.now()
                self.last_run_status = "SUCCESS"
                self.last_run_message = f"Completed in {duration:.2f} seconds"
                self.last_run_duration = duration
                
                logging.info(f"Task completed: {self.name} in {duration:.2f} seconds")
                return result
            except Exception as e:
                logging.error(f"Error in task {self.name} (attempt {attempt}/{self.max_retries}): {str(e)}")
                logging.error(traceback.format_exc())
                
                if attempt < self.max_retries:
                    logging.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    self.last_run_time = datetime.now()
                    self.last_run_status = "FAILED"
                    self.last_run_message = str(e)
                    self.last_run_duration = duration
                    
                    logging.error(f"Task {self.name} failed after {self.max_retries} attempts")
                    raise
    
    def execute(self):
        """Implement in subclasses to execute the task."""
        raise NotImplementedError("Subclasses must implement execute()")

    def get_status(self):
        """Get the task's execution status."""
        if not self.last_run_time:
            return {"name": self.name, "status": "NEVER_RUN", "message": "Never executed"}
        
        return {
            "name": self.name,
            "status": self.last_run_status,
            "last_run": self.last_run_time.isoformat(),
            "duration": f"{self.last_run_duration:.2f}s",
            "message": self.last_run_message
        }


class FetchPlayerStatsTask(PipelineTask):
    """Task to fetch player game stats using getPlayerGameStats.py"""
    
    def __init__(self, season='2024'):
        super().__init__("fetch_player_stats")
        self.season = season
    
    def execute(self):
        script_path = SCRIPTS_DIR / 'getPlayerGameStats.py'
        
        # Run the script to fetch player stats
        result = subprocess.run([
            sys.executable, str(script_path),
            '--seasons', self.season,
            '--max-workers', '10'
        ], capture_output=True, text=True, check=True)
        
        logging.info(f"Fetch player stats output: {result.stdout}")
        
        # Return the count of new games from the output
        for line in result.stdout.split('\n'):
            if "- Added" in line and "new games" in line:
                try:
                    count = int(line.split("Added")[1].split("new")[0].strip())
                    return {"new_games_added": count}
                except (ValueError, IndexError):
                    pass
        
        return {"new_games_added": 0}


class FetchTeamRostersTask(PipelineTask):
    """Task to fetch team rosters"""
    
    def execute(self):
        script_path = SCRIPTS_DIR / 'getTeamRoster.py'
        
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, check=True)
        
        logging.info(f"Fetch team rosters output: {result.stdout}")
        return {"status": "completed"}


class FetchUpcomingGamesTask(PipelineTask):
    """Task to fetch upcoming NBA games"""
    
    def execute(self):
        script_path = SCRIPTS_DIR / 'getUpcomingGames.py'
        
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, check=True)
        
        logging.info(f"Fetch upcoming games output: {result.stdout}")
        return {"status": "completed"}


class CleanStatsTask(PipelineTask):
    """Task to clean player stats data"""
    
    def execute(self):
        script_path = SCRIPTS_DIR / 'cleanStats.py'
        
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, check=True)
        
        logging.info(f"Clean stats output: {result.stdout}")
        
        # Check if the clean stats database was created
        if not os.path.exists(CLEAN_STATS_DB):
            raise RuntimeError(f"Clean stats database not created at {CLEAN_STATS_DB}")
        
        # Split data into train/test sets
        self._split_train_test_data()
        
        return {"status": "completed"}
    
    def _split_train_test_data(self):
        """Split the clean stats data into training and testing sets"""
        with db_lock:
            # Connect to the clean stats database
            conn = sqlite3.connect(CLEAN_STATS_DB)
            clean_df = pd.read_sql_query("SELECT * FROM clean_player_stats", conn)
            conn.close()
            
            # Sort by date
            clean_df['game_date'] = pd.to_datetime(clean_df['game_date'])
            clean_df = clean_df.sort_values('game_date')
            
            # Use the most recent 20% of data for testing
            split_idx = int(len(clean_df) * 0.8)
            train_df = clean_df.iloc[:split_idx]
            test_df = clean_df.iloc[split_idx:]
            
            # Save to separate databases
            train_conn = sqlite3.connect(TRAIN_DB)
            train_df.to_sql('clean_player_stats', train_conn, if_exists='replace', index=False)
            train_conn.close()
            
            test_conn = sqlite3.connect(TEST_DB)
            test_df.to_sql('clean_player_stats', test_conn, if_exists='replace', index=False)
            test_conn.close()
            
            logging.info(f"Split data into {len(train_df)} training examples and {len(test_df)} testing examples")


class EnhanceFeaturesTask(PipelineTask):
    """Task to create enhanced features for modeling"""
    
    def execute(self):
        script_path = MODELS_DIR / 'enhance_features.py'
        
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, check=True)
        
        logging.info(f"Enhance features output: {result.stdout}")
        
        # Check if the enhanced stats database was created
        if not os.path.exists(ENHANCED_STATS_DB):
            raise RuntimeError(f"Enhanced stats database not created at {ENHANCED_STATS_DB}")
        
        return {"status": "completed"}


class FetchProjectionsTask(PipelineTask):
    """Task to fetch and store projections"""
    
    def execute(self):
        script_path = PRIZEPICKS_DIR / 'getProjections.py'
        
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, check=True)
        
        logging.info(f"Fetch projections output: {result.stdout}")
        
        # Store projections in the centralized database
        script_path = PRIZEPICKS_DIR / 'store_projections.py'
        
        result = subprocess.run([
            sys.executable, str(script_path)
        ], capture_output=True, text=True, check=True)
        
        logging.info(f"Store projections output: {result.stdout}")
        
        # Parse the number of projections from the output
        for line in result.stdout.split('\n'):
            if "Database now contains" in line and "projections" in line:
                try:
                    count = int(line.split("contains")[1].split("projections")[0].strip())
                    return {"projections_count": count}
                except (ValueError, IndexError):
                    pass
        
        return {"projections_count": 0}


class TrainModelsTask(PipelineTask):
    """Task to train prediction models"""
    
    def execute(self):
        script_path = MODELS_DIR / 'player_predictions.py'
        
        # Train models with enhanced features and ensemble models
        result = subprocess.run([
            sys.executable, str(script_path),
            '--train-db', str(TRAIN_DB),
            '--test-db', str(TEST_DB),
            '--enhanced-db', str(ENHANCED_STATS_DB)
        ], capture_output=True, text=True, check=True)
        
        logging.info(f"Train models output: {result.stdout}")
        
        # Parse evaluation metrics from the output
        metrics = {}
        lines = result.stdout.split('\n')
        eval_start_idx = -1
        
        # Find where model evaluation starts
        for i, line in enumerate(lines):
            if "Model Evaluation Results:" in line:
                eval_start_idx = i + 1
                break
        
        if eval_start_idx > 0 and eval_start_idx < len(lines):
            # Try to parse the table headers and values
            try:
                # Assuming headers are in the format "Metric  RMSE  MAE  R²  Avg Value  Error %  Model"
                metrics_data = []
                header_line = None
                data_started = False
                
                for line in lines[eval_start_idx:]:
                    if "Metric" in line and "RMSE" in line and "Error %" in line:
                        header_line = line
                        data_started = True
                        continue
                    
                    if data_started and line.strip():
                        metrics_data.append(line)
                    
                    # Stop when we hit the next section
                    if data_started and not line.strip():
                        break
                
                # Process metrics if we found them
                if header_line and metrics_data:
                    metrics["evaluation"] = metrics_data
            except Exception as e:
                logging.warning(f"Could not parse evaluation metrics: {e}")
        
        return {
            "status": "completed",
            "metrics": metrics
        }


class GeneratePredictionsTask(PipelineTask):
    """Task to generate predictions and compare with projections"""
    
    def execute(self):
        script_path = MODELS_DIR / 'player_predictions.py'
        
        # Generate predictions and compare with projections
        result = subprocess.run([
            sys.executable, str(script_path),
            '--compare-projections',
            '--export-projections-csv', str(DASHBOARD_DIR / 'latest_projections.csv'),
            '--dashboard-data', str(DASHBOARD_DIR),
            '--detailed-confidence'
        ], capture_output=True, text=True, check=True)
        
        logging.info(f"Generate predictions output: {result.stdout}")
        
        # Also export top predictions to CSV
        result = subprocess.run([
            sys.executable, str(script_path),
            '--export-csv', str(DASHBOARD_DIR / 'top_predictions.csv')
        ], capture_output=True, text=True, check=True)
        
        # Store historical predictions for tracking
        self._store_historical_predictions()
        
        return {"status": "completed"}
    
    def _store_historical_predictions(self):
        """Store today's predictions in historical database for performance tracking"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with db_lock:
            # Create historical database if it doesn't exist
            conn = sqlite3.connect(HISTORICAL_DB)
            cursor = conn.cursor()
            
            # Create historical predictions table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT,
                team TEXT,
                opponent TEXT, 
                stat_type TEXT,
                line_score REAL,
                our_prediction REAL,
                prediction TEXT,
                edge REAL,
                edge_percent REAL,
                confidence REAL,
                prediction_date DATE,
                result TEXT,
                actual_value REAL,
                correct BOOLEAN,
                verified_date DATE
            )
            ''')
            
            # Create historical model performance table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric TEXT,
                rmse REAL,
                mae REAL,
                r2 REAL,
                avg_value REAL,
                error_pct REAL,
                model_type TEXT,
                evaluation_date DATE
            )
            ''')
            
            # Import the latest projections comparison results
            try:
                latest_projections = pd.read_csv(DASHBOARD_DIR / 'latest_projections.csv')
                
                # Add prediction date and leave result fields empty for now
                latest_projections['prediction_date'] = today
                latest_projections['result'] = None
                latest_projections['actual_value'] = None
                latest_projections['correct'] = None
                latest_projections['verified_date'] = None
                
                # Insert into historical database
                latest_projections.to_sql('historical_predictions', conn, if_exists='append', 
                                        index=False, 
                                        if_exists_constraints=['player_name', 'stat_type', 'prediction_date'])
                
                logging.info(f"Stored {len(latest_projections)} predictions in historical database")
                
            except Exception as e:
                logging.error(f"Error storing historical predictions: {e}")
            
            # Import the latest model performance metrics
            try:
                model_eval_path = DASHBOARD_DIR / 'model_evaluation.json'
                if os.path.exists(model_eval_path):
                    with open(model_eval_path, 'r') as f:
                        model_metrics = json.load(f)
                    
                    # Convert to DataFrame
                    metrics_df = pd.DataFrame(model_metrics)
                    metrics_df['evaluation_date'] = today
                    
                    # Insert into historical database
                    metrics_df.to_sql('historical_model_performance', conn, if_exists='append', 
                                    index=False)
                    
                    logging.info(f"Stored {len(metrics_df)} model metrics in historical database")
                
            except Exception as e:
                logging.error(f"Error storing historical model metrics: {e}")
            
            conn.commit()
            conn.close()


class GenerateDashboardTask(PipelineTask):
    """Task to generate dashboard data and track historical performance"""
    
    def execute(self):
        # Create dashboard index file if it doesn't exist
        dashboard_html = NBA_DIR / 'dashboard.html'
        if not os.path.exists(dashboard_html):
            logging.warning(f"Dashboard HTML file not found at {dashboard_html}")
        
        # Update dashboard data timestamp file
        timestamp_path = DASHBOARD_DIR / 'last_updated.json'
        timestamp_data = {
            "last_updated": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        with open(timestamp_path, 'w') as f:
            json.dump(timestamp_data, f)
        
        logging.info(f"Updated dashboard timestamp at {timestamp_path}")
        
        # Generate historical performance metrics
        self._generate_historical_metrics()
        
        return {"status": "completed"}
    
    def _generate_historical_metrics(self):
        """Generate historical performance metrics for dashboard"""
        with db_lock:
            # Connect to historical database
            if not os.path.exists(HISTORICAL_DB):
                logging.warning(f"Historical database not found at {HISTORICAL_DB}")
                return
            
            conn = sqlite3.connect(HISTORICAL_DB)
            
            # Get verified predictions (where we know the actual result)
            verified_predictions = pd.read_sql_query(
                "SELECT * FROM historical_predictions WHERE verified_date IS NOT NULL", 
                conn
            )
            
            # Calculate accuracy metrics
            if len(verified_predictions) > 0:
                # Overall accuracy
                overall_accuracy = verified_predictions['correct'].mean()
                
                # Accuracy by confidence tier
                high_conf = verified_predictions[verified_predictions['confidence'] >= 0.8]
                med_conf = verified_predictions[(verified_predictions['confidence'] >= 0.7) & 
                                              (verified_predictions['confidence'] < 0.8)]
                low_conf = verified_predictions[(verified_predictions['confidence'] >= 0.6) & 
                                              (verified_predictions['confidence'] < 0.7)]
                
                high_conf_accuracy = high_conf['correct'].mean() if len(high_conf) > 0 else 0
                med_conf_accuracy = med_conf['correct'].mean() if len(med_conf) > 0 else 0
                low_conf_accuracy = low_conf['correct'].mean() if len(low_conf) > 0 else 0
                
                # Accuracy by stat type
                stat_accuracy = verified_predictions.groupby('stat_type')['correct'].mean().to_dict()
                
                # Accuracy over time
                time_accuracy = verified_predictions.groupby('prediction_date')['correct'].mean().to_dict()
                
                # Create metrics JSON
                metrics = {
                    "overall_accuracy": overall_accuracy,
                    "accuracy_by_confidence": {
                        "high_conf": high_conf_accuracy,
                        "med_conf": med_conf_accuracy,
                        "low_conf": low_conf_accuracy
                    },
                    "samples_by_confidence": {
                        "high_conf": len(high_conf),
                        "med_conf": len(med_conf),
                        "low_conf": len(low_conf)
                    },
                    "accuracy_by_stat_type": stat_accuracy,
                    "accuracy_over_time": time_accuracy,
                    "total_verified_predictions": len(verified_predictions),
                    "last_updated": datetime.now().isoformat()
                }
                
                # Save metrics to JSON
                metrics_path = DASHBOARD_DIR / 'historical_performance.json'
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                logging.info(f"Generated historical performance metrics with {len(verified_predictions)} verified predictions")
            else:
                logging.info("No verified predictions found for historical metrics")
            
            conn.close()


class VerifyPredictionsTask(PipelineTask):
    """Task to verify prediction results against actual game outcomes"""
    
    def execute(self):
        script_path = SCRIPTS_DIR / 'get_game_results.py'
        
        # Run the verification script with a 7-day lookback window
        result = subprocess.run([
            sys.executable, str(script_path),
            '--days', '7'
        ], capture_output=True, text=True, check=True)
        
        logging.info(f"Verification script output: {result.stdout}")
        
        # Parse verification stats from the output
        stats = {}
        for line in result.stdout.split('\n'):
            if "Processed:" in line:
                try:
                    stats["processed"] = int(line.split(':')[1].strip().split(' ')[0])
                except (ValueError, IndexError):
                    pass
            elif "Found games:" in line:
                try:
                    stats["found_games"] = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "Not found games:" in line:
                try:
                    stats["not_found_games"] = int(line.split(':')[1].strip())
                except (ValueError, IndexError):
                    pass
            elif "Verified:" in line:
                try:
                    stats["verified_count"] = int(line.split(':')[1].strip().split(' ')[0])
                except (ValueError, IndexError):
                    pass
            elif "Accuracy:" in line:
                try:
                    # Extract accuracy percentage and correct/total counts
                    parts = line.split(':')[1].strip().split('(')
                    accuracy = float(parts[0].replace('%', '').strip())
                    counts = parts[1].replace(')', '').split('/')
                    correct = int(counts[0])
                    total = int(counts[1])
                    
                    stats["accuracy"] = accuracy
                    stats["correct_predictions"] = correct
                    stats["total_verified"] = total
                except (ValueError, IndexError):
                    pass
        
        # Check if we successfully verified any predictions
        if "verified_count" in stats and stats["verified_count"] > 0:
            # Generate updated historical performance metrics
            generate_dashboard_task = GenerateDashboardTask()
            generate_dashboard_task.run()
            
            return {
                "status": "completed", 
                "verified_count": stats["verified_count"],
                "accuracy": stats.get("accuracy", 0)
            }
        else:
            return {"status": "completed", "verified_count": 0}


class PipelineOrchestrator:
    """Orchestrates the entire prediction pipeline."""
    
    def __init__(self):
        self.tasks = {
            "fetch_player_stats": FetchPlayerStatsTask(),
            "fetch_team_rosters": FetchTeamRostersTask(),
            "fetch_upcoming_games": FetchUpcomingGamesTask(),
            "clean_stats": CleanStatsTask(),
            "enhance_features": EnhanceFeaturesTask(),
            "fetch_projections": FetchProjectionsTask(),
            "train_models": TrainModelsTask(),
            "generate_predictions": GeneratePredictionsTask(),
            "generate_dashboard": GenerateDashboardTask(),
            "verify_predictions": VerifyPredictionsTask()
        }
        
        self.task_dependencies = {
            "clean_stats": ["fetch_player_stats", "fetch_team_rosters"],
            "enhance_features": ["clean_stats"],
            "train_models": ["enhance_features"],
            "generate_predictions": ["train_models", "fetch_projections"],
            "generate_dashboard": ["generate_predictions"],
            "verify_predictions": []  # Independent task
        }
        
        self.task_status = {}
    
    def run_task(self, task_name):
        """Run a single task by name."""
        if task_name not in self.tasks:
            logging.error(f"Task {task_name} not found")
            return False
        
        # Check dependencies
        if task_name in self.task_dependencies:
            for dependency in self.task_dependencies[task_name]:
                if dependency not in self.task_status or self.task_status[dependency].get("status") != "SUCCESS":
                    logging.error(f"Cannot run {task_name}: dependency {dependency} not satisfied")
                    return False
        
        # Run the task
        task = self.tasks[task_name]
        try:
            result = task.run()
            self.task_status[task_name] = {
                "status": "SUCCESS",
                "last_run": datetime.now().isoformat(),
                "result": result
            }
            return True
        except Exception as e:
            self.task_status[task_name] = {
                "status": "FAILED",
                "last_run": datetime.now().isoformat(),
                "error": str(e)
            }
            return False
    
    def run_pipeline(self, tasks=None):
        """Run the complete pipeline or specified tasks."""
        if not tasks:
            # Run all tasks in dependency order
            tasks = [
                "fetch_player_stats",
                "fetch_team_rosters",
                "fetch_upcoming_games",
                "clean_stats",
                "enhance_features", 
                "fetch_projections",
                "train_models",
                "generate_predictions",
                "generate_dashboard",
                "verify_predictions"
            ]
        
        logging.info(f"Starting pipeline with tasks: {', '.join(tasks)}")
        
        # Track overall status
        succeeded = 0
        failed = 0
        
        for task_name in tasks:
            if task_name not in self.tasks:
                logging.warning(f"Task {task_name} not found, skipping")
                continue
            
            logging.info(f"Running task: {task_name}")
            if self.run_task(task_name):
                succeeded += 1
            else:
                failed += 1
        
        logging.info(f"Pipeline completed. Tasks: {succeeded} succeeded, {failed} failed")
        return succeeded, failed
    
    def get_pipeline_status(self):
        """Get the status of all pipeline tasks."""
        status = {}
        for task_name, task in self.tasks.items():
            status[task_name] = task.get_status()
        return status
    
    def schedule_tasks(self):
        """Set up the scheduling for regular task runs."""
        # Schedule player stats and upcoming games update daily
        schedule.every().day.at("04:00").do(self.run_task, "fetch_player_stats")
        schedule.every().day.at("04:30").do(self.run_task, "fetch_team_rosters")
        schedule.every().day.at("05:00").do(self.run_task, "fetch_upcoming_games")
        
        # Schedule data processing daily
        schedule.every().day.at("06:00").do(self.run_task, "clean_stats")
        schedule.every().day.at("06:30").do(self.run_task, "enhance_features")
        
        # Schedule model training weekly
        schedule.every().monday.at("07:00").do(self.run_task, "train_models")
        
        # Schedule projections and predictions multiple times per day
        schedule.every(3).hours.do(self.run_task, "fetch_projections")
        schedule.every(3).hours.do(self.run_task, "generate_predictions")
        schedule.every(3).hours.do(self.run_task, "generate_dashboard")
        
        # Schedule prediction verification daily
        schedule.every().day.at("08:00").do(self.run_task, "verify_predictions")
        
        # Schedule full pipeline refresh weekly
        schedule.every().sunday.at("23:00").do(self.run_pipeline)
        
        logging.info("Task scheduling initialized")
    
    def run_scheduler(self):
        """Run the scheduler in a loop."""
        self.schedule_tasks()
        
        logging.info("Starting scheduler...")
        while True:
            schedule.run_pending()
            time.sleep(60)


def create_directory_structure():
    """Create the necessary directory structure for the pipeline."""
    directories = [
        DATA_DIR,
        LOGS_DIR,
        DASHBOARD_DIR
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        logging.info(f"Created directory: {directory}")


def main():
    """Main function to run the automated pipeline."""
    parser = argparse.ArgumentParser(description='NBA Predictions Automated Pipeline')
    parser.add_argument('--task', type=str, help='Run a specific task')
    parser.add_argument('--pipeline', action='store_true', help='Run the complete pipeline')
    parser.add_argument('--schedule', action='store_true', help='Start the scheduler')
    parser.add_argument('--status', action='store_true', help='Show task status')
    args = parser.parse_args()
    
    # Create directory structure
    create_directory_structure()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator()
    
    if args.task:
        logging.info(f"Running task: {args.task}")
        success = orchestrator.run_task(args.task)
        if success:
            print(f"Task {args.task} completed successfully.")
        else:
            print(f"Task {args.task} failed.")
            sys.exit(1)
    
    elif args.pipeline:
        logging.info("Running complete pipeline")
        succeeded, failed = orchestrator.run_pipeline()
        print(f"Pipeline completed. Tasks: {succeeded} succeeded, {failed} failed.")
        if failed > 0:
            sys.exit(1)
    
    elif args.schedule:
        logging.info("Starting scheduler")
        orchestrator.run_scheduler()
    
    elif args.status:
        status = orchestrator.get_pipeline_status()
        print("\nPipeline Task Status:")
        print("=====================")
        for task, task_status in status.items():
            status_str = task_status["status"]
            status_color = ""
            if status_str == "SUCCESS":
                status_color = "\033[92m"  # Green
            elif status_str == "FAILED":
                status_color = "\033[91m"  # Red
            elif status_str == "NEVER_RUN":
                status_color = "\033[93m"  # Yellow
            
            reset_color = "\033[0m"
            
            print(f"{task}: {status_color}{status_str}{reset_color}")
            if "last_run" in task_status:
                print(f"  Last run: {task_status['last_run']}")
            if "duration" in task_status:
                print(f"  Duration: {task_status['duration']}")
            if "message" in task_status:
                print(f"  Message: {task_status['message']}")
            print("")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()