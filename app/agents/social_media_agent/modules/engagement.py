"""
Instagram Engagement Automation Module
Handles likes, comments, follows, unfollows, and DMs
"""
import logging
import time
import random
from typing import Optional, Dict, List, Any
from instagrapi import Client

logger = logging.getLogger(__name__)


class InstagramEngagement:
    """Handles Instagram engagement automation with safety features"""
    
    def __init__(self, client: Client):
        """
        Initialize engagement handler
        
        Args:
            client: Authenticated Instagrapi Client instance
        """
        self.client = client
        self.action_delays = {
            'like': (2, 5),
            'comment': (5, 10),
            'follow': (3, 7),
            'unfollow': (3, 7),
            'dm': (10, 20)
        }
    
    def _safe_delay(self, action_type: str = 'like'):
        """
        Add safe delay between actions
        
        Args:
            action_type: Type of action for appropriate delay
        """
        min_delay, max_delay = self.action_delays.get(action_type, (2, 5))
        delay = random.uniform(min_delay, max_delay)
        logger.debug(f"Waiting {delay:.2f} seconds before next action...")
        time.sleep(delay)
    
    def like_media(self, media_id: str) -> Dict[str, Any]:
        """
        Like a media post
        
        Args:
            media_id: Media ID to like
            
        Returns:
            Dict with like result
        """
        try:
            logger.info(f"Liking media: {media_id}")
            
            result = self.client.media_like(media_id)
            self._safe_delay('like')
            
            return {
                "success": True,
                "media_id": media_id,
                "action": "like"
            }
            
        except Exception as e:
            logger.error(f"Failed to like media: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def unlike_media(self, media_id: str) -> Dict[str, Any]:
        """
        Unlike a media post
        
        Args:
            media_id: Media ID to unlike
            
        Returns:
            Dict with unlike result
        """
        try:
            logger.info(f"Unliking media: {media_id}")
            
            result = self.client.media_unlike(media_id)
            self._safe_delay('like')
            
            return {
                "success": True,
                "media_id": media_id,
                "action": "unlike"
            }
            
        except Exception as e:
            logger.error(f"Failed to unlike media: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def comment_on_media(self, media_id: str, text: str) -> Dict[str, Any]:
        """
        Comment on a media post
        
        Args:
            media_id: Media ID to comment on
            text: Comment text
            
        Returns:
            Dict with comment result
        """
        try:
            logger.info(f"Commenting on media: {media_id}")
            
            comment = self.client.media_comment(media_id, text)
            self._safe_delay('comment')
            
            return {
                "success": True,
                "media_id": media_id,
                "comment_id": comment.pk,
                "text": text,
                "action": "comment"
            }
            
        except Exception as e:
            logger.error(f"Failed to comment: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_comment(self, media_id: str, comment_id: str) -> Dict[str, Any]:
        """
        Delete a comment
        
        Args:
            media_id: Media ID
            comment_id: Comment ID to delete
            
        Returns:
            Dict with deletion result
        """
        try:
            logger.info(f"Deleting comment: {comment_id}")
            
            result = self.client.comment_delete(comment_id)
            
            return {
                "success": True,
                "comment_id": comment_id,
                "action": "delete_comment"
            }
            
        except Exception as e:
            logger.error(f"Failed to delete comment: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def follow_user(self, user_id: str) -> Dict[str, Any]:
        """
        Follow a user
        
        Args:
            user_id: User ID to follow
            
        Returns:
            Dict with follow result
        """
        try:
            logger.info(f"Following user: {user_id}")
            
            result = self.client.user_follow(user_id)
            self._safe_delay('follow')
            
            return {
                "success": True,
                "user_id": user_id,
                "action": "follow"
            }
            
        except Exception as e:
            logger.error(f"Failed to follow user: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def unfollow_user(self, user_id: str) -> Dict[str, Any]:
        """
        Unfollow a user
        
        Args:
            user_id: User ID to unfollow
            
        Returns:
            Dict with unfollow result
        """
        try:
            logger.info(f"Unfollowing user: {user_id}")
            
            result = self.client.user_unfollow(user_id)
            self._safe_delay('unfollow')
            
            return {
                "success": True,
                "user_id": user_id,
                "action": "unfollow"
            }
            
        except Exception as e:
            logger.error(f"Failed to unfollow user: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def send_dm(self, user_ids: List[str], text: str) -> Dict[str, Any]:
        """
        Send a direct message
        
        Args:
            user_ids: List of user IDs to send message to
            text: Message text
            
        Returns:
            Dict with DM result
        """
        try:
            logger.info(f"Sending DM to {len(user_ids)} users")
            
            thread = self.client.direct_send(text, user_ids)
            self._safe_delay('dm')
            
            return {
                "success": True,
                "thread_id": thread.id,
                "recipients": user_ids,
                "text": text,
                "action": "send_dm"
            }
            
        except Exception as e:
            logger.error(f"Failed to send DM: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def send_dm_photo(self, user_ids: List[str], photo_path: str, text: str = "") -> Dict[str, Any]:
        """
        Send a photo via direct message
        
        Args:
            user_ids: List of user IDs
            photo_path: Path to photo file
            text: Optional message text
            
        Returns:
            Dict with DM result
        """
        try:
            logger.info(f"Sending photo DM to {len(user_ids)} users")
            
            thread = self.client.direct_send_photo(photo_path, user_ids, text)
            self._safe_delay('dm')
            
            return {
                "success": True,
                "thread_id": thread.id,
                "recipients": user_ids,
                "media_type": "photo",
                "action": "send_dm_photo"
            }
            
        except Exception as e:
            logger.error(f"Failed to send photo DM: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def send_dm_video(self, user_ids: List[str], video_path: str, text: str = "") -> Dict[str, Any]:
        """
        Send a video via direct message
        
        Args:
            user_ids: List of user IDs
            video_path: Path to video file
            text: Optional message text
            
        Returns:
            Dict with DM result
        """
        try:
            logger.info(f"Sending video DM to {len(user_ids)} users")
            
            thread = self.client.direct_send_video(video_path, user_ids, text)
            self._safe_delay('dm')
            
            return {
                "success": True,
                "thread_id": thread.id,
                "recipients": user_ids,
                "media_type": "video",
                "action": "send_dm_video"
            }
            
        except Exception as e:
            logger.error(f"Failed to send video DM: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def bulk_like_user_posts(
        self,
        username: str,
        amount: int = 5,
        max_amount: int = 10
    ) -> Dict[str, Any]:
        """
        Like multiple posts from a user (with safety limits)
        
        Args:
            username: Target username
            amount: Number of posts to like
            max_amount: Maximum allowed (safety limit)
            
        Returns:
            Dict with results
        """
        try:
            # Safety check
            if amount > max_amount:
                amount = max_amount
                logger.warning(f"Amount capped at {max_amount} for safety")
            
            logger.info(f"Liking {amount} posts from {username}")
            
            # Get user posts
            user_id = self.client.user_id_from_username(username)
            medias = self.client.user_medias(user_id, amount=amount)
            
            results = []
            for media in medias:
                result = self.like_media(media.pk)
                results.append(result)
                
                if not result['success']:
                    logger.warning(f"Failed to like {media.pk}, stopping bulk like")
                    break
            
            successful = sum(1 for r in results if r['success'])
            
            return {
                "success": True,
                "username": username,
                "total_attempted": len(results),
                "successful_likes": successful,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Bulk like failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def bulk_comment_on_hashtag(
        self,
        hashtag: str,
        comment_texts: List[str],
        amount: int = 3,
        max_amount: int = 5
    ) -> Dict[str, Any]:
        """
        Comment on posts from a hashtag feed
        
        Args:
            hashtag: Target hashtag
            comment_texts: List of comment variations
            amount: Number of posts to comment on
            max_amount: Maximum allowed (safety limit)
            
        Returns:
            Dict with results
        """
        try:
            # Safety check
            if amount > max_amount:
                amount = max_amount
                logger.warning(f"Amount capped at {max_amount} for safety")
            
            hashtag = hashtag.strip('#')
            logger.info(f"Commenting on {amount} posts from #{hashtag}")
            
            # Get hashtag posts
            medias = self.client.hashtag_medias_recent(hashtag, amount=amount)
            
            results = []
            for i, media in enumerate(medias):
                # Rotate through comment texts
                comment_text = comment_texts[i % len(comment_texts)]
                
                result = self.comment_on_media(media.pk, comment_text)
                results.append(result)
                
                if not result['success']:
                    logger.warning(f"Failed to comment on {media.pk}, stopping")
                    break
            
            successful = sum(1 for r in results if r['success'])
            
            return {
                "success": True,
                "hashtag": hashtag,
                "total_attempted": len(results),
                "successful_comments": successful,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Bulk comment failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def engage_with_user(
        self,
        username: str,
        like_posts: int = 3,
        comment_texts: Optional[List[str]] = None,
        follow: bool = False
    ) -> Dict[str, Any]:
        """
        Comprehensive engagement with a user
        
        Args:
            username: Target username
            like_posts: Number of posts to like
            comment_texts: Optional list of comments to post
            follow: Whether to follow the user
            
        Returns:
            Dict with engagement results
        """
        try:
            logger.info(f"Engaging with user: {username}")
            
            user_id = self.client.user_id_from_username(username)
            results = {
                "success": True,
                "username": username,
                "actions": []
            }
            
            # Like posts
            if like_posts > 0:
                like_result = self.bulk_like_user_posts(username, amount=like_posts)
                results["actions"].append({
                    "type": "likes",
                    "result": like_result
                })
            
            # Comment on posts
            if comment_texts and len(comment_texts) > 0:
                medias = self.client.user_medias(user_id, amount=min(len(comment_texts), 2))
                
                for i, media in enumerate(medias):
                    if i < len(comment_texts):
                        comment_result = self.comment_on_media(media.pk, comment_texts[i])
                        results["actions"].append({
                            "type": "comment",
                            "result": comment_result
                        })
            
            # Follow user
            if follow:
                follow_result = self.follow_user(user_id)
                results["actions"].append({
                    "type": "follow",
                    "result": follow_result
                })
            
            logger.info(f"Engagement completed with {username}")
            return results
            
        except Exception as e:
            logger.error(f"User engagement failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def engage_with_hashtag_audience(
        self,
        hashtag: str,
        users_to_engage: int = 5,
        likes_per_user: int = 2,
        follow_users: bool = False
    ) -> Dict[str, Any]:
        """
        Engage with users who posted on a hashtag
        
        Args:
            hashtag: Target hashtag
            users_to_engage: Number of users to engage with
            likes_per_user: Posts to like per user
            follow_users: Whether to follow engaged users
            
        Returns:
            Dict with engagement results
        """
        try:
            hashtag = hashtag.strip('#')
            logger.info(f"Engaging with {users_to_engage} users from #{hashtag}")
            
            # Get posts from hashtag
            medias = self.client.hashtag_medias_recent(hashtag, amount=users_to_engage)
            
            # Get unique users
            unique_users = {}
            for media in medias:
                if media.user.username not in unique_users:
                    unique_users[media.user.username] = media.user.pk
                if len(unique_users) >= users_to_engage:
                    break
            
            results = []
            for username, user_id in unique_users.items():
                engagement_result = self.engage_with_user(
                    username,
                    like_posts=likes_per_user,
                    follow=follow_users
                )
                results.append(engagement_result)
                
                # Extra delay between users
                time.sleep(random.uniform(5, 10))
            
            return {
                "success": True,
                "hashtag": hashtag,
                "users_engaged": len(results),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Hashtag audience engagement failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
