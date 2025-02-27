import React, { useState, useEffect } from 'react';
import axios from 'axios';

function VideoList({ token }) {
  const [videos, setVideos] = useState();

  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const response = await axios.get('/api/videos', {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setVideos(response.data);
      } catch (error) {
        console.error('Error fetching videos:', error);
        // Handle error (e.g., display error message)
      }
    };

    fetchVideos();
  }, [token]);

  return (
    <div>
      <h2>Videos</h2>
      <ul>
        {videos.map((video) => (
          <li key={video.id}>
            <h3>{video.title}</h3>
            <p>{video.description}</p>
            {/* Add more video details or actions here */}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default VideoList;