import os
import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any, List

load_dotenv()

def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('trading.log'),
            logging.StreamHandler()
        ]
    )

class Config:
    """Configuration loader and manager"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        setup_logging(self.get('logging.level', 'INFO'))
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        # Replace environment variables in config
        config = self._replace_env_vars(config)
        return config
    
    def _replace_env_vars(self, config: Any) -> Any:
        """Recursively replace ${ENV_VAR} placeholders with environment variables"""
        if isinstance(config, dict):
            return {k: self._replace_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._replace_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, config)
        return config
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: config.get('trading.initial_capital', 1000)
        """
        keys = path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_trading_pairs(self) -> List[str]:
        """Get list of trading pairs"""
        return self.get('trading_pairs', [])
    
    def get_exchange_config(self) -> Dict[str, Any]:
        """Get exchange configuration"""
        return self.get('exchange', {})
    
    def get_trading_config(self) -> Dict[str, Any]:
        """Get trading configuration"""
        return self.get('trading', {})
    
    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration"""
        return self.get('risk', {})
    
    def get_backtesting_config(self) -> Dict[str, Any]:
        """Get backtesting configuration"""
        return self.get('backtesting', {})
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """Get strategy weights for combined strategy"""
        return self.get('strategy_weights', {})
    
    def get_notification_config(self) -> Dict[str, Any]:
        """Get notification configuration"""
        return self.get('notifications', {})

# Global config instance
_config = None

def get_config(config_path: str = "config.yaml") -> Config:
    """Get global config instance"""
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

# Environment variable getters
def get_email_password() -> str:
    """Get email password from environment"""
    return os.getenv('EMAIL_PASSWORD', '')

def get_coinbase_api_key() -> str:
    """Get Coinbase API key from environment"""
    return os.getenv('COINBASE_API_KEY', '')

def get_coinbase_secret() -> str:
    """Get Coinbase API secret from environment"""
    return os.getenv('COINBASE_SECRET', '')

def get_coinbase_passphrase() -> str:
    """Get Coinbase API passphrase from environment"""
    return os.getenv('COINBASE_PASSPHRASE', '')