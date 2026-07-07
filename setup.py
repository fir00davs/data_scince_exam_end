"""
Setup script to initialize the project
Run this after cloning: python setup.py
"""
import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories"""
    logger.info("Creating directories...")
    dirs = [
        'data/raw',
        'data/processed',
        'models',
        'notebooks',
        'bot',
        'src',
        'tests'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"  ✓ {dir_path}")


def create_env_file():
    """Create .env file from template"""
    logger.info("Setting up .env file...")
    
    if Path('.env').exists():
        logger.warning("  ⚠️  .env already exists, skipping")
        return
    
    if not Path('.env.example').exists():
        logger.error("  ✗ .env.example not found")
        return
    
    # Copy template
    with open('.env.example', 'r') as f:
        content = f.read()
    
    with open('.env', 'w') as f:
        f.write(content)
    
    logger.info("  ✓ Created .env from template")
    logger.warning("  ⚠️  Please edit .env and add your TELEGRAM_BOT_TOKEN")


def check_python_version():
    """Check Python version"""
    logger.info("Checking Python version...")
    
    if sys.version_info < (3, 8):
        logger.error(f"  ✗ Python 3.8+ required, you have {sys.version}")
        sys.exit(1)
    
    logger.info(f"  ✓ Python {sys.version.split()[0]}")


def install_dependencies():
    """Install dependencies"""
    logger.info("Installing dependencies...")
    
    if not Path('requirements.txt').exists():
        logger.error("  ✗ requirements.txt not found")
        return False
    
    # Check if in virtual environment
    if not hasattr(sys, 'real_prefix') and not (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    ):
        logger.warning("  ⚠️  Not in virtual environment")
        logger.warning("  Consider running: python -m venv venv")
    
    logger.info("  To install dependencies, run:")
    logger.info("    pip install -r requirements.txt")
    
    return True


def print_next_steps():
    """Print setup instructions"""
    logger.info("\n" + "=" * 60)
    logger.info("SETUP COMPLETE!")
    logger.info("=" * 60)
    
    steps = [
        ("1. Edit .env file", "Add your Telegram bot token from @BotFather"),
        ("2. Install dependencies", "pip install -r requirements.txt"),
        ("3. Download data", "python src/data_loader.py"),
        ("4. Train models", "python src/train.py"),
        ("5. Run bot", "python bot/main.py"),
    ]
    
    for step, desc in steps:
        logger.info(f"\n{step}")
        logger.info(f"  → {desc}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Documentation: See README.md for more information")
    logger.info("=" * 60)


def main():
    """Run setup"""
    logger.info("=" * 60)
    logger.info("Movie Recommendation Bot - Setup")
    logger.info("=" * 60 + "\n")
    
    try:
        check_python_version()
        create_directories()
        create_env_file()
        install_dependencies()
        print_next_steps()
        
        logger.info("\n✅ Setup completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"\n❌ Setup failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
