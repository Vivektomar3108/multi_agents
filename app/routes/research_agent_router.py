# # app/routes/research_agent_router.py

# from fastapi import APIRouter, UploadFile, File, Form, Depends, Request, Query
# from app.controllers.research_agent_controller import research_agent_controller
# from app.middleware.auth import get_current_user

# router = APIRouter(prefix="/research_agent", tags=["Research Agent"])

# @router.post("/chat")
# async def chat_endpoint(
#     request: Request,
#     user=Depends(get_current_user),

#     # API DOCS WILL SHOW THESE because they are explicit params:
#     chat_id: str = Form(..., description="Unique chat session ID"),
#     query: str = Form(..., description="User query text"),
#     file: UploadFile | None = File(None, description="Optional file upload"),

#     # Query param with docs support
#     stream: bool = Query(False, description="Enable streaming response")
# ):
#     return await research_agent_controller.chat(
#         request=request,
#         user=user,
#         chat_id=chat_id,
#         query=query,
#         file=file,
#         stream=stream
#     )
