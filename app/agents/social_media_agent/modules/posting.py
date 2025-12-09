"""
Instagram Posting Module
Handles all types of content posting (photos, videos, reels, stories)
"""
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
from instagrapi import Client
from instagrapi.types import Location, Usertag, StoryMention, StoryLink, StoryHashtag

logger = logging.getLogger(__name__)


class InstagramPosting:
    """Handles Instagram content posting"""
    
    def __init__(self, client: Client):
        """
        Initialize posting handler
        
        Args:
            client: Authenticated Instagrapi Client instance
        """
        self.client = client
    
    def upload_photo(
        self,
        photo_path: str,
        caption: str = "",
        hashtags: Optional[List[str]] = None,
        location: Optional[Location] = None,
        usertags: Optional[List[Usertag]] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload a photo to Instagram feed
        
        Args:
            photo_path: Path to photo file
            caption: Photo caption
            hashtags: List of hashtags (without #)
            location: Location object for geotagging
            usertags: List of user tags
            extra_data: Additional metadata (alt_text, etc.)
            
        Returns:
            Dict with upload result and media info
        """
        try:
            # Build full caption with hashtags
            full_caption = caption
            if hashtags:
                hashtag_text = " ".join([f"#{tag.strip('#')}" for tag in hashtags])
                full_caption = f"{caption}\n\n{hashtag_text}"
            
            logger.info(f"Uploading photo: {photo_path}")
            
            # Prepare extra data
            if extra_data is None:
                extra_data = {}
            
            # Upload photo
            media = self.client.photo_upload(
                path=photo_path,
                caption=full_caption,
                location=location,
                usertags=usertags,
                extra_data=extra_data
            )
            
            logger.info(f"Photo uploaded successfully. Media ID: {media.pk}")
            
            return {
                "success": True,
                "media_id": media.pk,
                "media_type": "photo",
                "code": media.code,
                "url": f"https://www.instagram.com/p/{media.code}/",
                "caption": full_caption
            }
            
        except Exception as e:
            logger.error(f"Photo upload failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_video(
        self,
        video_path: str,
        caption: str = "",
        hashtags: Optional[List[str]] = None,
        thumbnail: Optional[str] = None,
        location: Optional[Location] = None,
        usertags: Optional[List[Usertag]] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload a video to Instagram feed
        
        Args:
            video_path: Path to video file
            caption: Video caption
            hashtags: List of hashtags
            thumbnail: Path to thumbnail image
            location: Location object
            usertags: List of user tags
            extra_data: Additional metadata
            
        Returns:
            Dict with upload result and media info
        """
        try:
            # Build full caption
            full_caption = caption
            if hashtags:
                hashtag_text = " ".join([f"#{tag.strip('#')}" for tag in hashtags])
                full_caption = f"{caption}\n\n{hashtag_text}"
            
            logger.info(f"Uploading video: {video_path}")
            
            if extra_data is None:
                extra_data = {}
            
            # Upload video
            media = self.client.video_upload(
                path=video_path,
                caption=full_caption,
                thumbnail=thumbnail,
                location=location,
                usertags=usertags,
                extra_data=extra_data
            )
            
            logger.info(f"Video uploaded successfully. Media ID: {media.pk}")
            
            return {
                "success": True,
                "media_id": media.pk,
                "media_type": "video",
                "code": media.code,
                "url": f"https://www.instagram.com/p/{media.code}/",
                "caption": full_caption
            }
            
        except Exception as e:
            logger.error(f"Video upload failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_reel(
        self,
        video_path: str,
        caption: str = "",
        hashtags: Optional[List[str]] = None,
        thumbnail: Optional[str] = None,
        cover_frame_ts: Optional[float] = None,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Upload a reel to Instagram
        
        Args:
            video_path: Path to video file
            caption: Reel caption
            hashtags: List of hashtags
            thumbnail: Path to custom thumbnail
            cover_frame_ts: Timestamp for cover frame (in seconds)
            extra_data: Additional metadata
            
        Returns:
            Dict with upload result and media info
        """
        try:
            # Build full caption
            full_caption = caption
            if hashtags:
                hashtag_text = " ".join([f"#{tag.strip('#')}" for tag in hashtags])
                full_caption = f"{caption}\n\n{hashtag_text}"
            
            logger.info(f"Uploading reel: {video_path}")
            
            if extra_data is None:
                extra_data = {}
            
            # Upload reel (clip)
            media = self.client.clip_upload(
                path=video_path,
                caption=full_caption,
                thumbnail=thumbnail,
                extra_data=extra_data
            )
            
            logger.info(f"Reel uploaded successfully. Media ID: {media.pk}")
            
            return {
                "success": True,
                "media_id": media.pk,
                "media_type": "reel",
                "code": media.code,
                "url": f"https://www.instagram.com/reel/{media.code}/",
                "caption": full_caption
            }
            
        except Exception as e:
            logger.error(f"Reel upload failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_album(
        self,
        media_paths: List[str],
        caption: str = "",
        hashtags: Optional[List[str]] = None,
        location: Optional[Location] = None,
        usertags: Optional[List[Usertag]] = None
    ) -> Dict[str, Any]:
        """
        Upload an album (carousel) to Instagram
        
        Args:
            media_paths: List of photo/video paths (max 10)
            caption: Album caption
            hashtags: List of hashtags
            location: Location object
            usertags: List of user tags
            
        Returns:
            Dict with upload result and media info
        """
        try:
            if len(media_paths) > 10:
                raise ValueError("Maximum 10 media items allowed in album")
            
            # Build full caption
            full_caption = caption
            if hashtags:
                hashtag_text = " ".join([f"#{tag.strip('#')}" for tag in hashtags])
                full_caption = f"{caption}\n\n{hashtag_text}"
            
            logger.info(f"Uploading album with {len(media_paths)} items")
            
            # Upload album
            media = self.client.album_upload(
                paths=media_paths,
                caption=full_caption,
                location=location,
                usertags=usertags
            )
            
            logger.info(f"Album uploaded successfully. Media ID: {media.pk}")
            
            return {
                "success": True,
                "media_id": media.pk,
                "media_type": "album",
                "code": media.code,
                "url": f"https://www.instagram.com/p/{media.code}/",
                "caption": full_caption,
                "items_count": len(media_paths)
            }
            
        except Exception as e:
            logger.error(f"Album upload failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_story_photo(
        self,
        photo_path: str,
        caption: Optional[str] = None,
        mentions: Optional[List[StoryMention]] = None,
        locations: Optional[List[Location]] = None,
        links: Optional[List[StoryLink]] = None,
        hashtags: Optional[List[StoryHashtag]] = None
    ) -> Dict[str, Any]:
        """
        Upload a photo story to Instagram
        
        Args:
            photo_path: Path to photo file
            caption: Story caption
            mentions: List of user mentions
            locations: List of locations
            links: List of swipe-up links (requires verified account)
            hashtags: List of hashtag stickers
            
        Returns:
            Dict with upload result and story info
        """
        try:
            logger.info(f"Uploading photo story: {photo_path}")
            
            # Upload photo story
            story = self.client.photo_upload_to_story(
                path=photo_path,
                caption=caption,
                mentions=mentions,
                locations=locations,
                links=links,
                hashtags=hashtags
            )
            
            logger.info(f"Photo story uploaded successfully. Story ID: {story.pk}")
            
            return {
                "success": True,
                "story_id": story.pk,
                "media_type": "story_photo",
                "code": story.code
            }
            
        except Exception as e:
            logger.error(f"Photo story upload failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def upload_story_video(
        self,
        video_path: str,
        caption: Optional[str] = None,
        mentions: Optional[List[StoryMention]] = None,
        locations: Optional[List[Location]] = None,
        links: Optional[List[StoryLink]] = None,
        hashtags: Optional[List[StoryHashtag]] = None,
        thumbnail: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload a video story to Instagram
        
        Args:
            video_path: Path to video file
            caption: Story caption
            mentions: List of user mentions
            locations: List of locations
            links: List of swipe-up links
            hashtags: List of hashtag stickers
            thumbnail: Path to thumbnail image
            
        Returns:
            Dict with upload result and story info
        """
        try:
            logger.info(f"Uploading video story: {video_path}")
            
            # Upload video story
            story = self.client.video_upload_to_story(
                path=video_path,
                caption=caption,
                mentions=mentions,
                locations=locations,
                links=links,
                hashtags=hashtags,
                thumbnail=thumbnail
            )
            
            logger.info(f"Video story uploaded successfully. Story ID: {story.pk}")
            
            return {
                "success": True,
                "story_id": story.pk,
                "media_type": "story_video",
                "code": story.code
            }
            
        except Exception as e:
            logger.error(f"Video story upload failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_media(self, media_id: str) -> Dict[str, Any]:
        """
        Delete a media post
        
        Args:
            media_id: Media ID to delete
            
        Returns:
            Dict with deletion result
        """
        try:
            logger.info(f"Deleting media: {media_id}")
            result = self.client.media_delete(media_id)
            
            logger.info(f"Media deleted successfully")
            
            return {
                "success": True,
                "media_id": media_id
            }
            
        except Exception as e:
            logger.error(f"Media deletion failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def edit_caption(self, media_id: str, new_caption: str) -> Dict[str, Any]:
        """
        Edit caption of existing post
        
        Args:
            media_id: Media ID to edit
            new_caption: New caption text
            
        Returns:
            Dict with edit result
        """
        try:
            logger.info(f"Editing caption for media: {media_id}")
            result = self.client.media_edit(media_id, new_caption)
            
            logger.info(f"Caption edited successfully")
            
            return {
                "success": True,
                "media_id": media_id,
                "new_caption": new_caption
            }
            
        except Exception as e:
            logger.error(f"Caption edit failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
