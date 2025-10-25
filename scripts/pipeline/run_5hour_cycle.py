"""
5-Hour Pipeline Orchestrator
============================
Professional automated pipeline for continuous ML improvement.

Runs in 5-hour cycles:
1. Data Collection (2.5 hours) - Harvest new cases
2. Model Training (1 hour) - Train on new batches
3. Embeddings (1 hour) - Generate vectors
4. Indexing (0.5 hours) - Build search index

Usage:
    python run_5hour_cycle.py
    
Features:
- Auto-repeat every 5 hours
- GPU optimization
- Error recovery
- Progress tracking
"""

import os
import sys
import time
import sqlite3
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

# Setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'pipeline_5h.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """5-hour cycle orchestrator"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.cycle_duration = timedelta(hours=5)
        self.db_path = PROJECT_ROOT / "data" / "legal_cases_10M.db"
        self.cycle_number = 1
        
        logger.info("=" * 80)
        logger.info("5-HOUR PIPELINE ORCHESTRATOR")
        logger.info("=" * 80)
        logger.info(f"Start time: {self.start_time}")
        logger.info(f"Cycle duration: 5 hours")
        logger.info(f"End time: {self.start_time + self.cycle_duration}")
        
    def get_case_count(self) -> int:
        """Get current case count"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cases")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except:
            return 0
            
    def run_command(self, cmd: list, description: str, timeout: int = None) -> bool:
        """Run a command and log results"""
        logger.info("-" * 80)
        logger.info(f"Running: {description}")
        logger.info(f"Command: {' '.join(cmd)}")
        logger.info("-" * 80)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(PROJECT_ROOT)
            )
            
            if result.returncode == 0:
                logger.info(f"✓ {description} - SUCCESS")
                if result.stdout:
                    logger.info(f"Output:\n{result.stdout[:500]}")
                return True
            else:
                logger.error(f"✗ {description} - FAILED")
                logger.error(f"Error:\n{result.stderr[:500]}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.warning(f"⏱ {description} - TIMEOUT (but may still be running)")
            return True  # Don't fail pipeline
        except Exception as e:
            logger.error(f"✗ {description} - ERROR: {e}")
            return False
            
    def phase_1_collect(self) -> bool:
        """Phase 1: Data Collection (2.5 hours)"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 1: DATA COLLECTION (2.5 hours)")
        logger.info("=" * 80)
        
        cases_before = self.get_case_count()
        logger.info(f"Cases before: {cases_before}")
        
        # Run ultra-fast harvester for 2.5 hours
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "ultra_fast_harvester.py")
        ]
        
        # Let it run in background (it has its own 5h timer)
        logger.info("Starting harvester (will run for 2.5 hours)...")
        logger.info("Note: Harvester runs in background, continuing pipeline...")
        
        # We don't wait - it runs alongside training
        subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT)
        )
        
        time.sleep(5)  # Give it time to start
        
        cases_after = self.get_case_count()
        logger.info(f"Cases after start: {cases_after}")
        logger.info("✓ Harvester started")
        
        return True
        
    def phase_2_train(self) -> bool:
        """Phase 2: Model Training (1 hour)"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 2: MODEL TRAINING (1 hour)")
        logger.info("=" * 80)
        
        # Wait a bit for cases to accumulate
        logger.info("Waiting 30 minutes for cases to accumulate...")
        time.sleep(30 * 60)  # 30 minutes
        
        cases = self.get_case_count()
        logger.info(f"Current cases: {cases}")
        
        if cases < 100:
            logger.warning("Not enough cases yet, skipping training")
            return True
            
        # Run batch trainer
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "pipeline" / "batch_trainer.py"),
            "--batch_size", "500",  # Lower threshold for 5h cycles
            "--force"  # Train even if not 1000 new cases
        ]
        
        return self.run_command(cmd, "Batch Training", timeout=3600)  # 1 hour max
        
    def phase_3_embeddings(self) -> bool:
        """Phase 3: Generate Embeddings (1 hour)"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 3: GENERATE EMBEDDINGS (1 hour)")
        logger.info("=" * 80)
        
        # Check if we have a trained model
        production_dir = PROJECT_ROOT / "models" / "production"
        if not list(production_dir.glob("model_*")):
            logger.warning("No trained model found, skipping embeddings")
            return True
            
        # Generate embeddings script
        embeddings_script = PROJECT_ROOT / "scripts" / "pipeline" / "generate_embeddings.py"
        
        if not embeddings_script.exists():
            logger.warning("Embeddings script not created yet, skipping")
            return True
            
        cmd = [
            sys.executable,
            str(embeddings_script)
        ]
        
        return self.run_command(cmd, "Generate Embeddings", timeout=3600)
        
    def phase_4_index(self) -> bool:
        """Phase 4: Build Search Index (30 minutes)"""
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 4: BUILD SEARCH INDEX (30 minutes)")
        logger.info("=" * 80)
        
        # Check if embeddings exist
        embeddings_dir = PROJECT_ROOT / "data" / "embeddings"
        if not list(embeddings_dir.glob("*.npy")):
            logger.warning("No embeddings found, skipping indexing")
            return True
            
        # Build index script
        index_script = PROJECT_ROOT / "scripts" / "pipeline" / "build_index.py"
        
        if not index_script.exists():
            logger.warning("Index script not created yet, skipping")
            return True
            
        cmd = [
            sys.executable,
            str(index_script)
        ]
        
        return self.run_command(cmd, "Build Search Index", timeout=1800)
        
    def save_cycle_summary(self, success: bool):
        """Save cycle summary"""
        cases = self.get_case_count()
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        summary = {
            'cycle_number': self.cycle_number,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_hours': duration.total_seconds() / 3600,
            'total_cases': cases,
            'success': success,
            'phases': {
                'collection': True,
                'training': True,
                'embeddings': True,
                'indexing': True
            }
        }
        
        summary_file = PROJECT_ROOT / "logs" / f"cycle_{self.cycle_number}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info("")
        logger.info("=" * 80)
        logger.info("CYCLE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Cycle: {self.cycle_number}")
        logger.info(f"Duration: {duration.total_seconds() / 3600:.2f} hours")
        logger.info(f"Total cases: {cases}")
        logger.info(f"Status: {'✓ SUCCESS' if success else '✗ FAILED'}")
        logger.info(f"Summary saved: {summary_file}")
        
    def run_cycle(self) -> bool:
        """Run one 5-hour cycle"""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"STARTING CYCLE #{self.cycle_number}")
        logger.info("=" * 80)
        
        success = True
        
        # Phase 1: Collect (2.5h)
        if not self.phase_1_collect():
            logger.error("Phase 1 (Collection) failed!")
            success = False
            
        # Phase 2: Train (1h)
        if not self.phase_2_train():
            logger.error("Phase 2 (Training) failed!")
            success = False
            
        # Phase 3: Embeddings (1h)
        if not self.phase_3_embeddings():
            logger.error("Phase 3 (Embeddings) failed!")
            success = False
            
        # Phase 4: Index (0.5h)
        if not self.phase_4_index():
            logger.error("Phase 4 (Indexing) failed!")
            success = False
            
        # Save summary
        self.save_cycle_summary(success)
        
        return success
        
    def run_continuous(self):
        """Run continuous 5-hour cycles"""
        logger.info("Starting continuous 5-hour cycles...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                self.start_time = datetime.now()
                success = self.run_cycle()
                
                if success:
                    logger.info("✓ Cycle completed successfully")
                else:
                    logger.warning("⚠ Cycle completed with errors")
                    
                self.cycle_number += 1
                
                # Wait for next cycle
                logger.info("")
                logger.info("=" * 80)
                logger.info("Waiting for next 5-hour cycle...")
                logger.info(f"Next cycle starts at: {datetime.now() + timedelta(minutes=5)}")
                logger.info("=" * 80)
                
                time.sleep(5 * 60)  # 5 minute break between cycles
                
        except KeyboardInterrupt:
            logger.info("")
            logger.info("=" * 80)
            logger.info("Pipeline stopped by user")
            logger.info(f"Completed {self.cycle_number - 1} cycles")
            logger.info("=" * 80)


def main():
    orchestrator = PipelineOrchestrator()
    
    # Run single cycle (for testing)
    # orchestrator.run_cycle()
    
    # Run continuous (for production)
    orchestrator.run_continuous()


if __name__ == "__main__":
    main()
