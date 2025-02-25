#!/usr/bin/env python3
import schedule
import time
import logging
import sys
import os
from datetime import datetime
from store_projections import store_projections, get_nba_projections

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('projections_scheduler.log')
    ]
)

def fetch_and_store_projections():
    """Fetch and store NBA projections."""
    try:
        start_time = datetime.now()
        logging.info("Starting projection fetch and store job")
        
        # Get projections and store them
        projections = get_nba_projections()
        store_projections(projections)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logging.info(f"Completed projection fetch and store job in {duration:.2f} seconds")
        
    except Exception as e:
        logging.error(f"Error in projection fetch and store job: {str(e)}")

def main():
    """Main function to schedule and run the projections job."""
    logging.info("Starting projections scheduler")
    
    # Schedule the job to run every 5 minutes
    schedule.every(5).minutes.do(fetch_and_store_projections)
    
    # Run the job immediately once
    fetch_and_store_projections()
    
    # Keep the script running and execute scheduled jobs
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Scheduler stopped by user")
            break
        except Exception as e:
            logging.error(f"Error in scheduler loop: {str(e)}")
            # Wait a bit before retrying
            time.sleep(5)

if __name__ == "__main__":
    main()
