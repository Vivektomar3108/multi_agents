"""
Instagram Authentication Module
Handles login, session management, and security features
"""
import os
import json
import time
import logging
from typing import Optional, Dict
from pathlib import Path
from instagrapi import Client
from instagrapi.exceptions import (
    LoginRequired,
    ChallengeRequired,
    PleaseWaitFewMinutes,
    TwoFactorRequired,
    RecaptchaChallengeForm,
    FeedbackRequired,
    RateLimitError
)

logger = logging.getLogger(__name__)


class InstagramAuth:
    """Handles Instagram authentication and session management"""
    
    def __init__(self, session_dir: str = "sessions"):
        """
        Initialize Instagram authentication handler
        
        Args:
            session_dir: Directory to store session files
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(exist_ok=True)
        self.client = None
        self.username = None
        
    def _get_session_path(self, username: str) -> Path:
        """Get path to session file for username"""
        return self.session_dir / f"{username}_session.json"
    
    def login(
        self,
        username: str,
        password: str,
        verification_code: Optional[str] = None,
        use_session: bool = True
    ) -> Client:
        """
        Login to Instagram with session management
        
        Args:
            username: Instagram username
            password: Instagram password
            verification_code: 2FA code if required
            use_session: Whether to use saved session
            
        Returns:
            Authenticated Client instance
        """
        self.username = username
        self.client = Client()
        
        # Configure client settings for better security
        self.client.delay_range = [2, 5]  # Random delay between actions
        
        session_path = self._get_session_path(username)
        
        # Try to load existing session
        if use_session and session_path.exists():
            try:
                logger.info(f"Loading session for {username}")
                self.client.load_settings(session_path)
                self.client.login(username, password)
                
                # Verify session is valid
                self.client.get_timeline_feed()
                logger.info(f"Successfully logged in using saved session")
                return self.client
                
            except Exception as e:
                logger.warning(f"Failed to use saved session: {e}")
                logger.info("Attempting fresh login...")
        
        # Fresh login
        try:
            logger.info(f"Logging in as {username}")
            self.client.login(username, password, verification_code=verification_code)
            
            # Save session
            if use_session:
                self.save_session(username)
            
            logger.info(f"Successfully logged in as {username}")
            return self.client
            
        except TwoFactorRequired as e:
            logger.error("Two-factor authentication required")
            raise Exception("Please provide verification_code parameter for 2FA")
            
        except ChallengeRequired as e:
            logger.error("Challenge required - Instagram needs verification")
            # Handle challenge
            self._handle_challenge()
            return self.client
            
        except PleaseWaitFewMinutes as e:
            logger.error("Rate limited - please wait a few minutes")
            raise Exception("Instagram rate limit detected. Wait 10-15 minutes before retry")
            
        except RateLimitError as e:
            logger.error("Rate limit error")
            raise Exception("Too many requests. Wait before retrying")
            
        except Exception as e:
            logger.error(f"Login failed: {str(e)}")
            raise
    
    def _handle_challenge(self) -> bool:
        """
        Handle Instagram challenge (email/SMS verification)
        
        Returns:
            True if challenge handled successfully
        """
        try:
            # Get challenge choice (email or SMS)
            if hasattr(self.client, 'challenge_code_handler'):
                logger.info("Challenge detected - sending code...")
                
                # Try to get challenge via email (0) or SMS (1)
                choice = 0  # Email by default
                self.client.challenge_resolve(choice)
                
                logger.info("Challenge code sent. Please check your email/phone")
                logger.info("Use client.challenge_code_handler to submit code")
                
                # In production, you'd want to pause here and wait for user input
                # For now, just log the requirement
                return True
                
        except Exception as e:
            logger.error(f"Challenge handling failed: {e}")
            return False
    
    def submit_challenge_code(self, code: str) -> bool:
        """
        Submit verification code for challenge
        
        Args:
            code: Verification code from email/SMS
            
        Returns:
            True if successful
        """
        try:
            if self.client:
                self.client.challenge_code_handler(self.username, code)
                logger.info("Challenge code submitted successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to submit challenge code: {e}")
            return False
    
    def save_session(self, username: str) -> bool:
        """
        Save current session to file
        
        Args:
            username: Instagram username
            
        Returns:
            True if saved successfully
        """
        try:
            session_path = self._get_session_path(username)
            self.client.dump_settings(session_path)
            logger.info(f"Session saved to {session_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    
    def load_session(self, username: str) -> bool:
        """
        Load session from file
        
        Args:
            username: Instagram username
            
        Returns:
            True if loaded successfully
        """
        try:
            session_path = self._get_session_path(username)
            if session_path.exists():
                self.client = Client()
                self.client.load_settings(session_path)
                self.username = username
                logger.info(f"Session loaded from {session_path}")
                return True
            else:
                logger.warning(f"No session file found for {username}")
                return False
        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            return False
    
    def logout(self) -> bool:
        """
        Logout from Instagram
        
        Returns:
            True if logged out successfully
        """
        try:
            if self.client:
                self.client.logout()
                logger.info("Logged out successfully")
                return True
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    def is_authenticated(self) -> bool:
        """
        Check if client is authenticated
        
        Returns:
            True if authenticated
        """
        try:
            if self.client:
                # Try a simple API call
                self.client.get_timeline_feed()
                return True
        except:
            return False
        return False
    
    def get_client(self) -> Optional[Client]:
        """Get the authenticated client instance"""
        return self.client
    
    def safe_delay(self, min_seconds: int = 2, max_seconds: int = 5):
        """
        Add a safe delay between actions to avoid rate limits
        
        Args:
            min_seconds: Minimum delay in seconds
            max_seconds: Maximum delay in seconds
        """
        import random
        delay = random.uniform(min_seconds, max_seconds)
        logger.debug(f"Waiting {delay:.2f} seconds...")
        time.sleep(delay)
