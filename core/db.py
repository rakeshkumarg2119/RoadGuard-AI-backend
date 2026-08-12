"""
RoadGuard AI - MongoDB Connection

Handles the asynchronous MongoDB Atlas connection using Motor.
"""

import os
import logging
from typing import Optional

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from motor.motor_asyncio import AsyncIOMotorDatabase


# ---------------------------------------------------------
# Load environment variables from .env
# ---------------------------------------------------------

load_dotenv()


# ---------------------------------------------------------
# Logger
# ---------------------------------------------------------

logger = logging.getLogger("roadguard.db")


# ---------------------------------------------------------
# MongoDB globals
# ---------------------------------------------------------

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


# ---------------------------------------------------------
# Database name
# ---------------------------------------------------------

DB_NAME = os.getenv(
    "MONGODB_DB_NAME",
    "roadguard"
)


# ---------------------------------------------------------
# Connect to MongoDB
# ---------------------------------------------------------

async def connect_to_mongo() -> None:
    """
    Connect to MongoDB Atlas.

    Reads MONGODB_URI from the .env file,
    creates the Motor client, tests the connection,
    and ensures required indexes exist.
    """

    global _client, _db

    # Reload .env in case environment variables changed
    load_dotenv()

    uri = os.getenv("MONGODB_URI")

    if not uri:
        raise RuntimeError(
            "MONGODB_URI is not set. "
            "Create a .env file and add your MongoDB connection string."
        )

    logger.info("Connecting to MongoDB Atlas...")

    try:
        # Create asynchronous MongoDB client
        _client = AsyncIOMotorClient(
            uri,
            serverSelectionTimeoutMS=5000,
        )

        # Select database
        _db = _client[DB_NAME]

        # Test connection immediately
        await _client.admin.command("ping")

        logger.info(
            "Connected to MongoDB successfully. Database: %s",
            DB_NAME,
        )

        # Create required indexes
        await _ensure_indexes()

    except Exception:
        # Clean up if connection fails
        if _client is not None:
            _client.close()

        _client = None
        _db = None

        logger.exception("MongoDB connection failed.")

        raise


# ---------------------------------------------------------
# Close MongoDB connection
# ---------------------------------------------------------

async def close_mongo_connection() -> None:
    """
    Close the MongoDB connection.
    """

    global _client, _db

    if _client is not None:
        _client.close()

        logger.info(
            "MongoDB connection closed."
        )

    _client = None
    _db = None


# ---------------------------------------------------------
# Ensure MongoDB indexes
# ---------------------------------------------------------

async def _ensure_indexes() -> None:
    """
    Create the 2dsphere index used by GPS queries.

    This index is required for MongoDB geospatial queries
    such as $nearSphere.
    """

    if _db is None:
        raise RuntimeError(
            "MongoDB database is not initialized."
        )

    await _db.potholes.create_index(
        [("gps", "2dsphere")]
    )

    logger.info(
        "MongoDB 2dsphere index ensured on potholes.gps"
    )


# ---------------------------------------------------------
# Get database
# ---------------------------------------------------------

def get_database() -> AsyncIOMotorDatabase:
    """
    Return the active MongoDB database.
    """

    if _db is None:
        raise RuntimeError(
            "MongoDB is not connected. "
            "Call connect_to_mongo() during application startup."
        )

    return _db


# ---------------------------------------------------------
# Get potholes collection
# ---------------------------------------------------------

def get_potholes_collection():
    """
    Return the potholes MongoDB collection.
    """

    return get_database().potholes