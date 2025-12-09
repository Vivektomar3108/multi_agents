"""
Instagram Content Intelligence Module
Provides AI-driven insights for content optimization
"""
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from collections import Counter
import re

logger = logging.getLogger(__name__)


class ContentIntelligence:
    """Provides intelligent content recommendations and analysis"""
    
    def __init__(self, client):
        """
        Initialize content intelligence
        
        Args:
            client: Authenticated Instagrapi Client instance
        """
        self.client = client
    
    def suggest_best_posting_times(self, posts_count: int = 30) -> Dict[str, Any]:
        """
        Suggest best posting times based on historical performance
        
        Args:
            posts_count: Number of recent posts to analyze
            
        Returns:
            Dict with posting time recommendations
        """
        try:
            logger.info(f"Analyzing {posts_count} posts for best posting times")
            
            user_info = self.client.account_info()
            medias = self.client.user_medias(user_info.pk, amount=posts_count)
            
            if not medias:
                return {
                    "success": False,
                    "error": "No posts found to analyze"
                }
            
            # Analyze posting times and engagement
            hour_engagement = {}
            day_engagement = {}
            
            for media in medias:
                if not media.taken_at:
                    continue
                
                hour = media.taken_at.hour
                day = media.taken_at.strftime("%A")
                engagement = media.like_count + media.comment_count
                
                # Track by hour
                if hour not in hour_engagement:
                    hour_engagement[hour] = []
                hour_engagement[hour].append(engagement)
                
                # Track by day
                if day not in day_engagement:
                    day_engagement[day] = []
                day_engagement[day].append(engagement)
            
            # Calculate average engagement per hour
            best_hours = []
            for hour, engagements in hour_engagement.items():
                avg_engagement = sum(engagements) / len(engagements)
                best_hours.append({
                    "hour": hour,
                    "avg_engagement": avg_engagement,
                    "posts_count": len(engagements)
                })
            
            best_hours.sort(key=lambda x: x['avg_engagement'], reverse=True)
            
            # Calculate average engagement per day
            best_days = []
            for day, engagements in day_engagement.items():
                avg_engagement = sum(engagements) / len(engagements)
                best_days.append({
                    "day": day,
                    "avg_engagement": avg_engagement,
                    "posts_count": len(engagements)
                })
            
            best_days.sort(key=lambda x: x['avg_engagement'], reverse=True)
            
            recommendations = {
                "success": True,
                "analysis_period": f"Last {len(medias)} posts",
                "best_hours": best_hours[:3],  # Top 3 hours
                "best_days": best_days[:3],    # Top 3 days
                "recommendations": {
                    "optimal_hour": best_hours[0]['hour'] if best_hours else None,
                    "optimal_day": best_days[0]['day'] if best_days else None,
                    "suggested_posting_time": f"{best_days[0]['day']} at {best_hours[0]['hour']}:00" if best_hours and best_days else "Need more data"
                }
            }
            
            logger.info("Posting time analysis completed")
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to suggest posting times: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_hashtags(
        self,
        caption: Optional[str] = None,
        niche: Optional[str] = None,
        count: int = 30
    ) -> Dict[str, Any]:
        """
        Generate hashtag suggestions based on caption or niche
        
        Args:
            caption: Post caption to analyze
            niche: Content niche (e.g., "fitness", "travel", "food")
            count: Number of hashtags to suggest
            
        Returns:
            Dict with hashtag suggestions
        """
        try:
            logger.info("Generating hashtag suggestions")
            
            hashtags = []
            
            # Predefined hashtag pools for common niches
            niche_hashtags = {
                "fitness": [
                    "fitness", "gym", "workout", "fitfam", "training", "bodybuilding",
                    "fitnessmotivation", "health", "fit", "exercise", "muscle", "strength",
                    "cardio", "healthylifestyle", "gains", "gymlife", "motivation", 
                    "wellness", "fitlife", "instafit", "fitnessjourney", "strong",
                    "fitnessgoals", "fitspiration", "personaltrainer", "gymrat",
                    "workoutmotivation", "fitnesstransformation", "fitnessaddict", "healthyliving"
                ],
                "travel": [
                    "travel", "wanderlust", "travelgram", "instatravel", "adventure",
                    "explore", "traveling", "travelphotography", "vacation", "trip",
                    "tourism", "travelblogger", "nature", "traveler", "photooftheday",
                    "traveltheworld", "beautiful", "instagood", "worldplaces", "passportready",
                    "travellife", "traveladdict", "travelholic", "beautifuldestinations",
                    "exploring", "adventuretime", "traveldiaries", "globetrotter", "tourist", "landscape"
                ],
                "food": [
                    "food", "foodie", "foodporn", "instafood", "yummy", "delicious",
                    "foodstagram", "foodlover", "foodphotography", "dinner", "lunch",
                    "homemade", "tasty", "cooking", "chef", "recipe", "foodblogger",
                    "healthyfood", "breakfast", "eat", "restaurant", "foodgasm",
                    "cuisine", "foodpics", "yum", "dessert", "foodiesofinstagram",
                    "instagood", "hungry", "eats"
                ],
                "fashion": [
                    "fashion", "style", "ootd", "fashionblogger", "fashionista", "model",
                    "fashionstyle", "instafashion", "outfit", "shopping", "beauty",
                    "stylish", "fashionable", "instagood", "beautiful", "fashiongram",
                    "lookbook", "streetstyle", "fashionweek", "fashiondiaries", "clothing",
                    "fashionpost", "fashionaddict", "trendy", "moda", "fashionlover",
                    "outfitoftheday", "fashiondesigner", "instastyle", "lifestyle"
                ],
                "photography": [
                    "photography", "photo", "photographer", "photooftheday", "picoftheday",
                    "instagood", "beautiful", "art", "nature", "portrait", "landscape",
                    "photoshoot", "canon", "nikon", "naturephotography", "instaphoto",
                    "photographylovers", "travelphotography", "portraitphotography", "ig_photo",
                    "photos", "pics", "streetphotography", "photographyislife", "love",
                    "camera", "photographyeveryday", "photographysouls", "mobilephotography", "photographylover"
                ],
                "business": [
                    "business", "entrepreneur", "marketing", "success", "motivation",
                    "entrepreneurship", "money", "startup", "businessowner", "mindset",
                    "hustle", "smallbusiness", "businessman", "businesswoman", "goals",
                    "leadership", "businesslife", "finance", "branding", "businessmindset",
                    "strategy", "networking", "inspire", "businesstips", "sales",
                    "digital marketing", "socialmediamarketing", "growyourbusiness", "ceo", "startuplife"
                ]
            }
            
            # If niche provided, use niche-specific hashtags
            if niche:
                niche_lower = niche.lower()
                if niche_lower in niche_hashtags:
                    hashtags = niche_hashtags[niche_lower][:count]
                else:
                    # Generic hashtags if niche not found
                    hashtags = ["instagood", "photooftheday", "love", "beautiful", 
                               "happy", "instadaily", "nature", "art", "lifestyle"]
            
            # If caption provided, extract keywords and suggest related hashtags
            if caption:
                # Extract words from caption
                words = re.findall(r'\b\w+\b', caption.lower())
                
                # Filter out common words
                common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                               'to', 'for', 'of', 'with', 'is', 'was', 'be', 'have', 'has'}
                keywords = [w for w in words if w not in common_words and len(w) > 3]
                
                # Add keywords as hashtags
                for keyword in keywords[:10]:
                    if keyword not in hashtags:
                        hashtags.append(keyword)
            
            # Ensure we have enough hashtags
            if len(hashtags) < count:
                # Add generic popular hashtags
                generic = ["instagood", "photooftheday", "beautiful", "love", "happy",
                          "instadaily", "follow", "like", "picoftheday", "instamood",
                          "amazing", "instalike", "bestoftheday", "smile", "fun",
                          "friends", "followme", "style", "instacool", "life"]
                
                for tag in generic:
                    if tag not in hashtags and len(hashtags) < count:
                        hashtags.append(tag)
            
            # Format hashtags
            formatted_hashtags = [f"#{tag}" for tag in hashtags[:count]]
            
            result = {
                "success": True,
                "hashtag_count": len(formatted_hashtags),
                "hashtags": formatted_hashtags,
                "hashtag_string": " ".join(formatted_hashtags),
                "niche": niche,
                "tips": [
                    "Mix popular and niche-specific hashtags",
                    "Use 20-30 relevant hashtags per post",
                    "Place hashtags in first comment to keep caption clean",
                    "Update hashtags regularly to avoid being shadowbanned"
                ]
            }
            
            logger.info(f"Generated {len(formatted_hashtags)} hashtags")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate hashtags: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_engagement_rate(self, posts_count: int = 20) -> Dict[str, Any]:
        """
        Analyze overall engagement rate
        
        Args:
            posts_count: Number of recent posts to analyze
            
        Returns:
            Dict with engagement analysis
        """
        try:
            logger.info(f"Analyzing engagement rate for {posts_count} posts")
            
            user_info = self.client.account_info()
            user = self.client.user_info(user_info.pk)
            medias = self.client.user_medias(user_info.pk, amount=posts_count)
            
            if not medias:
                return {
                    "success": False,
                    "error": "No posts found to analyze"
                }
            
            total_likes = sum(m.like_count for m in medias)
            total_comments = sum(m.comment_count for m in medias)
            total_engagement = total_likes + total_comments
            
            # Calculate engagement rate
            # Engagement Rate = (Total Engagement / (Followers * Posts)) * 100
            if user.follower_count > 0:
                engagement_rate = (total_engagement / (user.follower_count * len(medias))) * 100
            else:
                engagement_rate = 0
            
            # Calculate averages
            avg_likes = total_likes / len(medias)
            avg_comments = total_comments / len(medias)
            avg_engagement = total_engagement / len(medias)
            
            # Determine engagement quality
            if engagement_rate >= 10:
                quality = "Excellent"
            elif engagement_rate >= 5:
                quality = "Good"
            elif engagement_rate >= 2:
                quality = "Average"
            elif engagement_rate >= 1:
                quality = "Below Average"
            else:
                quality = "Poor"
            
            analysis = {
                "success": True,
                "followers": user.follower_count,
                "posts_analyzed": len(medias),
                "total_likes": total_likes,
                "total_comments": total_comments,
                "total_engagement": total_engagement,
                "avg_likes_per_post": round(avg_likes, 2),
                "avg_comments_per_post": round(avg_comments, 2),
                "avg_engagement_per_post": round(avg_engagement, 2),
                "engagement_rate": round(engagement_rate, 2),
                "engagement_quality": quality,
                "recommendations": self._get_engagement_recommendations(engagement_rate)
            }
            
            logger.info("Engagement analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze engagement: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_engagement_recommendations(self, engagement_rate: float) -> List[str]:
        """Get recommendations based on engagement rate"""
        
        if engagement_rate >= 10:
            return [
                "Your engagement is excellent! Keep doing what you're doing",
                "Consider creating similar content to maintain high engagement",
                "Engage with your audience regularly to maintain momentum"
            ]
        elif engagement_rate >= 5:
            return [
                "Good engagement! Try posting more consistently",
                "Experiment with different content types (reels, carousels)",
                "Increase audience interaction through stories and polls"
            ]
        elif engagement_rate >= 2:
            return [
                "Improve content quality with better visuals",
                "Post at optimal times when your audience is active",
                "Use relevant hashtags to reach new audiences",
                "Engage with your followers by responding to comments"
            ]
        else:
            return [
                "Focus on creating high-quality, valuable content",
                "Post consistently (at least 3-4 times per week)",
                "Research and use trending hashtags in your niche",
                "Engage actively with your target audience",
                "Collaborate with similar accounts for cross-promotion",
                "Use Instagram Stories and Reels to boost visibility"
            ]
    
    def recommend_content_ideas(self, niche: str) -> Dict[str, Any]:
        """
        Recommend content ideas based on niche
        
        Args:
            niche: Content niche
            
        Returns:
            Dict with content recommendations
        """
        try:
            logger.info(f"Generating content ideas for niche: {niche}")
            
            # Content idea templates by niche
            content_ideas = {
                "fitness": [
                    "Before/After transformation photos",
                    "Quick workout routines (60 seconds or less)",
                    "Healthy meal prep ideas",
                    "Motivational quotes over workout clips",
                    "Exercise form tutorials",
                    "Fitness challenges for followers",
                    "Day in the life of a fitness journey",
                    "Common workout mistakes to avoid",
                    "Progress tracking posts",
                    "Q&A about fitness and nutrition"
                ],
                "travel": [
                    "Hidden gems in popular destinations",
                    "Travel budget tips and hacks",
                    "Packing list essentials",
                    "Local food experiences",
                    "Sunrise/sunset locations",
                    "Travel itinerary breakdowns",
                    "Cultural experiences and traditions",
                    "Travel fails and funny moments",
                    "Comparison posts (expectations vs reality)",
                    "Behind-the-scenes of travel photography"
                ],
                "food": [
                    "Recipe videos (step-by-step)",
                    "Food plating and presentation",
                    "Cooking hacks and tips",
                    "Ingredient spotlight posts",
                    "Restaurant reviews and recommendations",
                    "Meal prep for the week",
                    "Seasonal recipe collections",
                    "Healthy alternatives to popular dishes",
                    "Cooking mistakes and how to fix them",
                    "Food photography tips"
                ],
                "fashion": [
                    "Outfit of the day (OOTD)",
                    "Styling one item multiple ways",
                    "Seasonal wardrobe essentials",
                    "Fashion trends breakdown",
                    "Affordable fashion finds",
                    "Outfit planning for specific occasions",
                    "Closet organization tips",
                    "Mix and match combinations",
                    "Fashion do's and don'ts",
                    "Sustainable fashion choices"
                ],
                "business": [
                    "Productivity tips and tools",
                    "Business growth strategies",
                    "Success stories and case studies",
                    "Common mistakes to avoid",
                    "Daily routines of successful entrepreneurs",
                    "Book recommendations for business",
                    "Motivational content and quotes",
                    "Industry news and trends",
                    "Behind-the-scenes of your business",
                    "Client testimonials and results"
                ]
            }
            
            niche_lower = niche.lower()
            ideas = content_ideas.get(niche_lower, [
                "Educational posts teaching something valuable",
                "Behind-the-scenes content",
                "User-generated content features",
                "Polls and interactive stories",
                "Trending topics in your industry",
                "Expert tips and tricks",
                "Motivational and inspirational posts",
                "Product or service showcases",
                "Customer success stories",
                "Q&A sessions with your audience"
            ])
            
            result = {
                "success": True,
                "niche": niche,
                "content_ideas": ideas,
                "general_tips": [
                    "Create content pillars (3-5 main themes)",
                    "Mix educational, entertaining, and promotional content (80/20 rule)",
                    "Use Instagram Reels for maximum reach",
                    "Post carousels for higher engagement",
                    "Repurpose top-performing content",
                    "Stay consistent with your brand aesthetic"
                ]
            }
            
            logger.info(f"Generated {len(ideas)} content ideas")
            return result
            
        except Exception as e:
            logger.error(f"Failed to recommend content: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_competitor(self, username: str) -> Dict[str, Any]:
        """
        Analyze a competitor's account
        
        Args:
            username: Competitor's username
            
        Returns:
            Dict with competitor analysis
        """
        try:
            logger.info(f"Analyzing competitor: {username}")
            
            user_id = self.client.user_id_from_username(username)
            user = self.client.user_info(user_id)
            medias = self.client.user_medias(user_id, amount=20)
            
            if not medias:
                return {
                    "success": False,
                    "error": "No posts found for analysis"
                }
            
            # Calculate metrics
            total_likes = sum(m.like_count for m in medias)
            total_comments = sum(m.comment_count for m in medias)
            avg_likes = total_likes / len(medias)
            avg_comments = total_comments / len(medias)
            
            # Engagement rate
            engagement_rate = ((total_likes + total_comments) / (user.follower_count * len(medias))) * 100 if user.follower_count > 0 else 0
            
            # Posting frequency
            if len(medias) >= 2:
                dates = [m.taken_at for m in medias if m.taken_at]
                if len(dates) >= 2:
                    date_diff = (dates[0] - dates[-1]).days
                    posts_per_week = (len(dates) / date_diff) * 7 if date_diff > 0 else 0
                else:
                    posts_per_week = 0
            else:
                posts_per_week = 0
            
            # Extract common hashtags
            all_hashtags = []
            for media in medias:
                if media.caption_text:
                    hashtags = re.findall(r'#(\w+)', media.caption_text)
                    all_hashtags.extend(hashtags)
            
            common_hashtags = []
            if all_hashtags:
                hashtag_counts = Counter(all_hashtags)
                common_hashtags = [f"#{tag}" for tag, count in hashtag_counts.most_common(10)]
            
            analysis = {
                "success": True,
                "username": username,
                "followers": user.follower_count,
                "following": user.following_count,
                "total_posts": user.media_count,
                "posts_analyzed": len(medias),
                "avg_likes": round(avg_likes, 2),
                "avg_comments": round(avg_comments, 2),
                "engagement_rate": round(engagement_rate, 2),
                "posts_per_week": round(posts_per_week, 2),
                "common_hashtags": common_hashtags,
                "is_verified": user.is_verified,
                "is_business": user.is_business,
                "insights": [
                    f"Posts approximately {round(posts_per_week, 1)} times per week",
                    f"Average engagement rate of {round(engagement_rate, 2)}%",
                    f"Gets around {round(avg_likes)} likes and {round(avg_comments)} comments per post"
                ]
            }
            
            logger.info("Competitor analysis completed")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze competitor: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
