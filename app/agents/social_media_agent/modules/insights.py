"""
Instagram Insights & Analytics Module
Fetches performance metrics for profile, posts, reels, and stories
"""
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from instagrapi import Client
from instagrapi.types import Media

logger = logging.getLogger(__name__)


class InstagramInsights:
    """Handles Instagram insights and analytics"""
    
    def __init__(self, client: Client):
        """
        Initialize insights handler
        
        Args:
            client: Authenticated Instagrapi Client instance
        """
        self.client = client
    
    def get_account_insights(self) -> Dict[str, Any]:
        """
        Get overall account insights
        
        Returns:
            Dict with account metrics
        """
        try:
            logger.info("Fetching account insights")
            
            # Get account info
            user_info = self.client.account_info()
            
            # Get user by username to get public metrics
            user = self.client.user_info(user_info.pk)
            
            insights = {
                "success": True,
                "username": user.username,
                "full_name": user.full_name,
                "biography": user.biography,
                "followers_count": user.follower_count,
                "following_count": user.following_count,
                "media_count": user.media_count,
                "is_verified": user.is_verified,
                "is_private": user.is_private,
                "is_business": user.is_business,
                "category": user.category,
                "timestamp": datetime.now().isoformat()
            }
            
            # Try to get business account insights (requires business account)
            try:
                # Get insights from recent media
                medias = self.client.user_medias(user_info.pk, amount=20)
                
                if medias:
                    total_likes = sum(m.like_count for m in medias)
                    total_comments = sum(m.comment_count for m in medias)
                    total_views = sum(m.view_count or 0 for m in medias if hasattr(m, 'view_count'))
                    
                    insights.update({
                        "recent_posts_count": len(medias),
                        "avg_likes": total_likes / len(medias) if medias else 0,
                        "avg_comments": total_comments / len(medias) if medias else 0,
                        "avg_views": total_views / len(medias) if total_views > 0 else 0,
                        "engagement_rate": (total_likes + total_comments) / (user.follower_count * len(medias)) * 100 if user.follower_count > 0 else 0
                    })
            except Exception as e:
                logger.warning(f"Could not fetch detailed insights: {e}")
            
            logger.info("Account insights retrieved successfully")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get account insights: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_media_insights(self, media_id: str) -> Dict[str, Any]:
        """
        Get insights for a specific post/reel
        
        Args:
            media_id: Media ID to analyze
            
        Returns:
            Dict with media insights
        """
        try:
            logger.info(f"Fetching insights for media: {media_id}")
            
            # Get media info
            media = self.client.media_info(media_id)
            
            insights = {
                "success": True,
                "media_id": media.pk,
                "media_type": str(media.media_type),
                "code": media.code,
                "url": f"https://www.instagram.com/p/{media.code}/",
                "caption": media.caption_text if media.caption_text else "",
                "taken_at": media.taken_at.isoformat() if media.taken_at else None,
                "like_count": media.like_count,
                "comment_count": media.comment_count,
                "play_count": media.play_count if hasattr(media, 'play_count') else 0,
                "view_count": media.view_count if hasattr(media, 'view_count') else 0,
                "is_paid_partnership": media.is_paid_partnership if hasattr(media, 'is_paid_partnership') else False,
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculate engagement rate
            if media.like_count or media.comment_count:
                user_info = self.client.user_info(media.user.pk)
                if user_info.follower_count > 0:
                    insights["engagement_rate"] = (
                        (media.like_count + media.comment_count) / user_info.follower_count * 100
                    )
            
            # Get likers if available
            try:
                likers = self.client.media_likers(media_id)
                insights["likers_count"] = len(likers)
            except:
                pass
            
            logger.info("Media insights retrieved successfully")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get media insights: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_story_insights(self, story_id: str) -> Dict[str, Any]:
        """
        Get insights for a specific story
        
        Args:
            story_id: Story media ID
            
        Returns:
            Dict with story insights
        """
        try:
            logger.info(f"Fetching insights for story: {story_id}")
            
            # Get story info
            story = self.client.story_info(story_id)
            
            insights = {
                "success": True,
                "story_id": story.pk,
                "media_type": str(story.media_type),
                "taken_at": story.taken_at.isoformat() if story.taken_at else None,
                "view_count": story.view_count if hasattr(story, 'view_count') else 0,
                "timestamp": datetime.now().isoformat()
            }
            
            # Get story viewers
            try:
                viewers = self.client.story_viewers(story_id)
                insights["viewers_count"] = len(viewers)
                insights["viewers"] = [
                    {
                        "username": v.username,
                        "full_name": v.full_name
                    } for v in viewers[:10]  # First 10 viewers
                ]
            except Exception as e:
                logger.warning(f"Could not fetch story viewers: {e}")
            
            logger.info("Story insights retrieved successfully")
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get story insights: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_media_insights_batch(self, media_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get insights for multiple media items
        
        Args:
            media_ids: List of media IDs
            
        Returns:
            List of insight dicts
        """
        results = []
        for media_id in media_ids:
            result = self.get_media_insights(media_id)
            results.append(result)
        
        return results
    
    def get_recent_posts_performance(self, count: int = 12) -> Dict[str, Any]:
        """
        Analyze performance of recent posts
        
        Args:
            count: Number of recent posts to analyze
            
        Returns:
            Dict with performance summary
        """
        try:
            logger.info(f"Analyzing {count} recent posts")
            
            user_info = self.client.account_info()
            medias = self.client.user_medias(user_info.pk, amount=count)
            
            if not medias:
                return {
                    "success": False,
                    "error": "No posts found"
                }
            
            # Calculate aggregate metrics
            total_likes = sum(m.like_count for m in medias)
            total_comments = sum(m.comment_count for m in medias)
            total_views = sum(m.view_count or 0 for m in medias if hasattr(m, 'view_count'))
            
            # Find best performing post
            best_post = max(medias, key=lambda m: m.like_count + m.comment_count)
            
            # Analyze posting times
            post_hours = [m.taken_at.hour for m in medias if m.taken_at]
            most_common_hour = max(set(post_hours), key=post_hours.count) if post_hours else None
            
            performance = {
                "success": True,
                "posts_analyzed": len(medias),
                "total_likes": total_likes,
                "total_comments": total_comments,
                "total_views": total_views,
                "avg_likes": total_likes / len(medias),
                "avg_comments": total_comments / len(medias),
                "avg_views": total_views / len(medias) if total_views > 0 else 0,
                "best_performing_post": {
                    "code": best_post.code,
                    "url": f"https://www.instagram.com/p/{best_post.code}/",
                    "likes": best_post.like_count,
                    "comments": best_post.comment_count,
                    "engagement": best_post.like_count + best_post.comment_count
                },
                "most_common_posting_hour": most_common_hour,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Performance analysis completed")
            return performance
            
        except Exception as e:
            logger.error(f"Failed to analyze performance: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_followers_growth(self, days: int = 7) -> Dict[str, Any]:
        """
        Track followers growth over time
        Note: This requires storing historical data as Instagram API doesn't provide historical follower counts
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with growth metrics
        """
        try:
            logger.info(f"Fetching followers growth for {days} days")
            
            user_info = self.client.account_info()
            user = self.client.user_info(user_info.pk)
            
            # Current snapshot
            growth = {
                "success": True,
                "current_followers": user.follower_count,
                "current_following": user.following_count,
                "timestamp": datetime.now().isoformat(),
                "note": "Historical data requires storing snapshots over time"
            }
            
            logger.info("Followers snapshot retrieved")
            return growth
            
        except Exception as e:
            logger.error(f"Failed to get followers growth: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_audience_demographics(self) -> Dict[str, Any]:
        """
        Get audience demographic information
        Note: Limited data available through public API
        
        Returns:
            Dict with audience info
        """
        try:
            logger.info("Fetching audience demographics")
            
            user_info = self.client.account_info()
            
            # Get sample of followers
            followers = self.client.user_followers(user_info.pk, amount=100)
            
            # Analyze sample
            verified_count = sum(1 for f in followers.values() if f.is_verified)
            business_count = sum(1 for f in followers.values() if f.is_business)
            private_count = sum(1 for f in followers.values() if f.is_private)
            
            demographics = {
                "success": True,
                "sample_size": len(followers),
                "verified_percentage": (verified_count / len(followers) * 100) if followers else 0,
                "business_accounts_percentage": (business_count / len(followers) * 100) if followers else 0,
                "private_accounts_percentage": (private_count / len(followers) * 100) if followers else 0,
                "timestamp": datetime.now().isoformat(),
                "note": "Based on sample of 100 followers"
            }
            
            logger.info("Audience demographics retrieved")
            return demographics
            
        except Exception as e:
            logger.error(f"Failed to get audience demographics: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
