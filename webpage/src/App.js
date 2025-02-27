
import React, { useState, useEffect } from 'react';
import './App.css';
import UploadForm from './components/UploadForm';
import LoginForm from './components/LoginForm';
import VideoList from './components/VideoList';
import PatientList from './components/PatientList'; // Add PatientList component
import { initializeApp } from 'firebase/app';
import { getAuth, onAuthStateChanged } from 'firebase/auth';
import axios from 'axios';

const firebaseConfig = {
  apiKey: "AIzaSyDBx-M3mB92ZekSgoVCDX_rqGGUy-Fi5_U",
  authDomain: "fitphysio-a6bdd.firebaseapp.com",
  databaseURL: "https://fitphysio-a6bdd-default-rtdb.firebaseio.com",
  projectId: "fitphysio-a6bdd",
  storageBucket: "fitphysio-a6bdd.firebasestorage.app",
  messagingSenderId: "379065816786",
  appId: "1:379065816786:web:1377392fafb70c48fe7f0b",
  measurementId: "G-TTLET9W2R5"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

function App() {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (authUser) => {
      if (authUser) {
        setUser(authUser);
        // Get the backend token after Firebase authentication
        const getBackendToken = async () => {
          try {
            const response = await axios.post('/api/login', {
              username: authUser.email,
              password: 'password', // Replace with actual password or handling
            });
            setToken(response.data.token);
          } catch (error) {
            console.error('Backend login error:', error);
          }
        };
        getBackendToken();
      } else {
        setUser(null);
        setToken(null);
      }
    });
    return () => unsubscribe();
  },);

  return (
    <div className="App">
      <h1>Video Analysis Dashboard</h1>
      {user ? (
        <div>
          <UploadForm token={token} />
          <VideoList token={token} />
          <PatientList token={token} /> {/* Add PatientList component */}
        </div>
      ) : (
        <LoginForm auth={auth} /> // Pass auth as a prop to LoginForm
      )}
    </div>
  );
}

export default App;