"""
Instagram Scraping & Data Extraction Module
Handles profile scraping, media downloading, and data collection
"""
import logging
from typing import Optional, Dict, List, Any
from pathlib import Path
from instagrapi import Client
from instagrapi.types import User, Media

logger = logging.getLogger(__name__)


class InstagramScraper:
    """Handles Instagram data scraping and extraction"""
    
    def __init__(self, client: Client):
        """
        Initialize scraper
        
        Args:
            client: Authenticated Instagrapi Client instance
        """
        self.client = client
    
    def get_user_profile(self, username: str) -> Dict[str, Any]:
        """
        Get detailed profile information for a user
        
        Args:
            username: Instagram username
            
        Returns:
            Dict with user profile data
        """
        try:
            logger.info(f"Fetching profile: {username}")
            
            # Get user ID from username
            user_id = self.client.user_id_from_username(username)
            
            # Get full user info
            user = self.client.user_info(user_id)
            
            profile = {
                "success": True,
                "user_id": user.pk,
                "username": user.username,
                "full_name": user.full_name,
                "biography": user.biography,
                "external_url": user.external_url,
                "followers_count": user.follower_count,
                "following_count": user.following_count,
                "media_count": user.media_count,
                "is_verified": user.is_verified,
                "is_private": user.is_private,
                "is_business": user.is_business,
                "category": user.category,
                "profile_pic_url": user.profile_pic_url,
                "public_email": user.public_email,
                "contact_phone_number": user.contact_phone_number,
                "business_category_name": user.business_category_name if hasattr(user, 'business_category_name') else None
            }
            
            logger.info(f"Profile retrieved: {username}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to get profile: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_user_followers(self, username: str, amount: int = 100) -> Dict[str, Any]:
        """
        Get followers list for a user
        
        Args:
            username: Instagram username
            amount: Number of followers to fetch
            
        Returns:
            Dict with followers list
        """
        try:
            logger.info(f"Fetching {amount} followers for: {username}")
            
            user_id = self.client.user_id_from_username(username)
            followers = self.client.user_followers(user_id, amount=amount)
            
            followers_list = [
                {
                    "user_id": user.pk,
                    "username": user.username,
                    "full_name": user.full_name,
                    "is_verified": user.is_verified,
                    "is_private": user.is_private,
                    "profile_pic_url": user.profile_pic_url
                }
                for user in followers.values()
            ]
            
            result = {
                "success": True,
                "username": username,
                "followers_count": len(followers_list),
                "followers": followers_list
            }
            
            logger.info(f"Retrieved {len(followers_list)} followers")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get followers: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_user_following(self, username: str, amount: int = 100) -> Dict[str, Any]:
        """
        Get following list for a user
        
        Args:
            username: Instagram username
            amount: Number of following to fetch
            
        Returns:
            Dict with following list
        """
        try:
            logger.info(f"Fetching {amount} following for: {username}")
            
            user_id = self.client.user_id_from_username(username)
            following = self.client.user_following(user_id, amount=amount)
            
            following_list = [
                {
                    "user_id": user.pk,
                    "username": user.username,
                    "full_name": user.full_name,
                    "is_verified": user.is_verified,
                    "is_private": user.is_private,
                    "profile_pic_url": user.profile_pic_url
                }
                for user in following.values()
            ]
            
            result = {
                "success": True,
                "username": username,
                "following_count": len(following_list),
                "following": following_list
            }
            
            logger.info(f"Retrieved {len(following_list)} following")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get following: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_user_posts(self, username: str, amount: int = 20) -> Dict[str, Any]:
        """
        Get posts from a user's profile
        
        Args:
            username: Instagram username
            amount: Number of posts to fetch
            
        Returns:
            Dict with posts list
        """
        try:
            logger.info(f"Fetching {amount} posts for: {username}")
            
            user_id = self.client.user_id_from_username(username)
            medias = self.client.user_medias(user_id, amount=amount)
            
            posts = [
                {
                    "media_id": media.pk,
                    "code": media.code,
                    "url": f"https://www.instagram.com/p/{media.code}/",
                    "media_type": str(media.media_type),
                    "caption": media.caption_text if media.caption_text else "",
                    "taken_at": media.taken_at.isoformat() if media.taken_at else None,
                    "like_count": media.like_count,
                    "comment_count": media.comment_count,
                    "thumbnail_url": media.thumbnail_url
                }
                for media in medias
            ]
            
            result = {
                "success": True,
                "username": username,
                "posts_count": len(posts),
                "posts": posts
            }
            
            logger.info(f"Retrieved {len(posts)} posts")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get posts: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_hashtag_feed(self, hashtag: str, amount: int = 20) -> Dict[str, Any]:
        """
        Get posts from a hashtag feed
        
        Args:
            hashtag: Hashtag to search (without #)
            amount: Number of posts to fetch
            
        Returns:
            Dict with hashtag posts
        """
        try:
            hashtag = hashtag.strip('#')
            logger.info(f"Fetching {amount} posts for hashtag: #{hashtag}")
            
            medias = self.client.hashtag_medias_recent(hashtag, amount=amount)
            
            posts = [
                {
                    "media_id": media.pk,
                    "code": media.code,
                    "url": f"https://www.instagram.com/p/{media.code}/",
                    "username": media.user.username,
                    "caption": media.caption_text if media.caption_text else "",
                    "like_count": media.like_count,
                    "comment_count": media.comment_count,
                    "taken_at": media.taken_at.isoformat() if media.taken_at else None
                }
                for media in medias
            ]
            
            result = {
                "success": True,
                "hashtag": hashtag,
                "posts_count": len(posts),
                "posts": posts
            }
            
            logger.info(f"Retrieved {len(posts)} posts from #{hashtag}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get hashtag feed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_location_feed(self, location_id: str, amount: int = 20) -> Dict[str, Any]:
        """
        Get posts from a location
        
        Args:
            location_id: Location ID
            amount: Number of posts to fetch
            
        Returns:
            Dict with location posts
        """
        try:
            logger.info(f"Fetching {amount} posts for location: {location_id}")
            
            medias = self.client.location_medias_recent(location_id, amount=amount)
            
            posts = [
                {
                    "media_id": media.pk,
                    "code": media.code,
                    "url": f"https://www.instagram.com/p/{media.code}/",
                    "username": media.user.username,
                    "caption": media.caption_text if media.caption_text else "",
                    "like_count": media.like_count,
                    "comment_count": media.comment_count
                }
                for media in medias
            ]
            
            result = {
                "success": True,
                "location_id": location_id,
                "posts_count": len(posts),
                "posts": posts
            }
            
            logger.info(f"Retrieved {len(posts)} posts from location")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get location feed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def download_photo(self, media_id: str, folder: str = "downloads") -> Dict[str, Any]:
        """
        Download a photo from Instagram
        
        Args:
            media_id: Media ID to download
            folder: Folder to save the photo
            
        Returns:
            Dict with download result and file path
        """
        try:
            logger.info(f"Downloading photo: {media_id}")
            
            folder_path = Path(folder)
            folder_path.mkdir(exist_ok=True)
            
            # Download photo
            file_path = self.client.photo_download(media_id, folder=folder)
            
            result = {
                "success": True,
                "media_id": media_id,
                "file_path": str(file_path),
                "media_type": "photo"
            }
            
            logger.info(f"Photo downloaded: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to download photo: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def download_video(self, media_id: str, folder: str = "downloads") -> Dict[str, Any]:
        """
        Download a video from Instagram
        
        Args:
            media_id: Media ID to download
            folder: Folder to save the video
            
        Returns:
            Dict with download result and file path
        """
        try:
            logger.info(f"Downloading video: {media_id}")
            
            folder_path = Path(folder)
            folder_path.mkdir(exist_ok=True)
            
            # Download video
            file_path = self.client.video_download(media_id, folder=folder)
            
            result = {
                "success": True,
                "media_id": media_id,
                "file_path": str(file_path),
                "media_type": "video"
            }
            
            logger.info(f"Video downloaded: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to download video: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def download_reel(self, media_id: str, folder: str = "downloads") -> Dict[str, Any]:
        """
        Download a reel from Instagram
        
        Args:
            media_id: Media ID to download
            folder: Folder to save the reel
            
        Returns:
            Dict with download result and file path
        """
        try:
            logger.info(f"Downloading reel: {media_id}")
            
            folder_path = Path(folder)
            folder_path.mkdir(exist_ok=True)
            
            # Download reel (clips)
            file_path = self.client.clip_download(media_id, folder=folder)
            
            result = {
                "success": True,
                "media_id": media_id,
                "file_path": str(file_path),
                "media_type": "reel"
            }
            
            logger.info(f"Reel downloaded: {file_path}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to download reel: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_media_comments(self, media_id: str, amount: int = 50) -> Dict[str, Any]:
        """
        Get comments from a media post
        
        Args:
            media_id: Media ID
            amount: Number of comments to fetch
            
        Returns:
            Dict with comments list
        """
        try:
            logger.info(f"Fetching {amount} comments for media: {media_id}")
            
            comments = self.client.media_comments(media_id, amount=amount)
            
            comments_list = [
                {
                    "comment_id": comment.pk,
                    "username": comment.user.username,
                    "text": comment.text,
                    "created_at": comment.created_at_utc.isoformat() if comment.created_at_utc else None,
                    "like_count": comment.like_count if hasattr(comment, 'like_count') else 0
                }
                for comment in comments
            ]
            
            result = {
                "success": True,
                "media_id": media_id,
                "comments_count": len(comments_list),
                "comments": comments_list
            }
            
            logger.info(f"Retrieved {len(comments_list)} comments")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get comments: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_media_likers(self, media_id: str) -> Dict[str, Any]:
        """
        Get list of users who liked a media post
        
        Args:
            media_id: Media ID
            
        Returns:
            Dict with likers list
        """
        try:
            logger.info(f"Fetching likers for media: {media_id}")
            
            likers = self.client.media_likers(media_id)
            
            likers_list = [
                {
                    "user_id": user.pk,
                    "username": user.username,
                    "full_name": user.full_name,
                    "is_verified": user.is_verified
                }
                for user in likers
            ]
            
            result = {
                "success": True,
                "media_id": media_id,
                "likers_count": len(likers_list),
                "likers": likers_list
            }
            
            logger.info(f"Retrieved {len(likers_list)} likers")
            return result
            
        except Exception as e:
            logger.error(f"Failed to get likers: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def search_users(self, query: str) -> Dict[str, Any]:
        """
        Search for users by username or name
        
        Args:
            query: Search query
            
        Returns:
            Dict with search results
        """
        try:
            logger.info(f"Searching users: {query}")
            
            users = self.client.search_users(query)
            
            users_list = [
                {
                    "user_id": user.pk,
                    "username": user.username,
                    "full_name": user.full_name,
                    "is_verified": user.is_verified,
                    "is_private": user.is_private,
                    "follower_count": user.follower_count
                }
                for user in users
            ]
            
            result = {
                "success": True,
                "query": query,
                "results_count": len(users_list),
                "users": users_list
            }
            
            logger.info(f"Found {len(users_list)} users")
            return result
            
        except Exception as e:
            logger.error(f"Failed to search users: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
