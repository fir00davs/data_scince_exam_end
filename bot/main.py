"""
Main Telegram Bot Application
"""
import logging
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, filters, ContextTypes
)

logger = logging.getLogger(__name__)

import config
from bot.model_loader import ModelLoader
from bot.handlers import BotHandlers


# Global model loader
model_loader = None
bot_handlers = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    try:
        response = bot_handlers.handle_start(update.effective_user.id)
        await update.message.reply_text(response, parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f"Error in start handler: {e}")
        await update.message.reply_text("❌ An error occurred. Please try again.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    try:
        response = bot_handlers.handle_help()
        await update.message.reply_text(response, parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f"Error in help handler: {e}")
        await update.message.reply_text("❌ An error occurred. Please try again.")


async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /search command"""
    try:
        if not context.args:
            await update.message.reply_text(
                "❌ Please provide a movie name\\. Example: `/search Inception`",
                parse_mode='MarkdownV2'
            )
            return
        
        query = ' '.join(context.args)
        response = bot_handlers.handle_search(update.effective_user.id, query)
        await update.message.reply_text(response, parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f"Error in search handler: {e}")
        await update.message.reply_text("❌ An error occurred. Please try again.")


async def rate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /rate command"""
    try:
        if len(context.args) < 2:
            await update.message.reply_text(
                "❌ Please use format: `/rate [movie_id] [1-5]`\\n"
                "Example: `/rate 100 5`",
                parse_mode='MarkdownV2'
            )
            return
        
        movie_id = context.args[0]
        rating = context.args[1]
        response = bot_handlers.handle_rate(update.effective_user.id, movie_id, rating)
        await update.message.reply_text(response, parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f"Error in rate handler: {e}")
        await update.message.reply_text("❌ An error occurred. Please try again.")


async def recommend(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /recommend command"""
    try:
        if not model_loader.is_ready():
            await update.message.reply_text(
                "❌ Models are not loaded\\. Please restart the bot\\.",
                parse_mode='MarkdownV2'
            )
            return
        
        response = bot_handlers.handle_recommend(update.effective_user.id)
        await update.message.reply_text(response, parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f"Error in recommend handler: {e}")
        await update.message.reply_text("❌ An error occurred. Please try again.")


async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stats command"""
    try:
        response = bot_handlers.handle_stats(update.effective_user.id)
        await update.message.reply_text(response, parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f"Error in stats handler: {e}")
        await update.message.reply_text("❌ An error occurred. Please try again.")


async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear command"""
    try:
        response = bot_handlers.handle_clear(update.effective_user.id)
        await update.message.reply_text(response, parse_mode='MarkdownV2')
    except Exception as e:
        logger.error(f"Error in clear handler: {e}")
        await update.message.reply_text("❌ An error occurred. Please try again.")


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(f"Exception while handling an update: {context.error}")
    
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ Sorry, something went wrong\\. Please try again later\\."
        )


def main():
    """Start the bot"""
    global model_loader, bot_handlers
    
    logger.info("=" * 60)
    logger.info("STARTING TELEGRAM BOT")
    logger.info("=" * 60)
    
    # Initialize model loader
    logger.info("Loading ML models...")
    model_loader = ModelLoader()
    
    if not model_loader.is_ready():
        logger.error("Failed to load models. Please run training first!")
        return
    
    logger.info("✓ Models loaded successfully")
    
    # Initialize handlers
    bot_handlers = BotHandlers(model_loader)
    
    # Create bot application
    token = config.TELEGRAM_BOT_TOKEN
    
    if token == "YOUR_BOT_TOKEN_HERE":
        logger.error("❌ Please set TELEGRAM_BOT_TOKEN in config.py or .env file!")
        logger.error("Get a token from @BotFather on Telegram")
        return
    
    app = Application.builder().token(token).build()
    
    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("search", search))
    app.add_handler(CommandHandler("rate", rate))
    app.add_handler(CommandHandler("recommend", recommend))
    app.add_handler(CommandHandler("stats", stats))
    app.add_handler(CommandHandler("clear", clear))
    
    # Add error handler
    app.add_error_handler(error_handler)
    
    logger.info("=" * 60)
    logger.info("BOT STARTED AND LISTENING")
    logger.info("=" * 60)
    logger.info("Press Ctrl+C to stop")
    
    # Start polling
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT
    )
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nBot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
