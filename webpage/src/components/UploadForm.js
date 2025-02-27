import React, { useState } from "react";
import axios from "axios";

function UploadForm({ token }) {
  const [videoFile, setVideoFile] = useState(null);
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");

  const handleFileChange = (e) => {
    setVideoFile(e.target.files);
  };

  const handleUpload = async () => {
    if (!videoFile) {
      alert("Please select a video file.");
      return;
    }

    try {
      const formData = new FormData();
      formData.append("title", title);
      formData.append("description", description);
      formData.append("video", videoFile);

      await axios.post("/api/upload", formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "multipart/form-data",
        },
      });

      setVideoFile(null);
      setTitle("");
      setDescription("");

      alert("Video uploaded successfully!");
    } catch (error) {
      console.error("Upload error:", error);
    }
  };

  return (
    <form>
      <div>
        <label htmlFor="title">Title:</label>
        <input
          type="text"
          id="title"
          value={title}
          onChange={(e) => setTitle(e.target.value)}
        />
      </div>
      <div>
        <label htmlFor="description">Description:</label>
        <textarea
          id="description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
        />
      </div>
      <div>
        <label htmlFor="video">Video:</label>
        <input
          type="file"
          id="video"
          accept="video/*"
          onChange={handleFileChange}
        />
      </div>
      <button type="button" onClick={handleUpload}>
        Upload
      </button>
    </form>
  );
}

export default UploadForm;