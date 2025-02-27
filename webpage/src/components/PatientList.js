import React, { useState, useEffect } from "react";
import axios from "axios";

function PatientList({ token }) {
  const [patients, setPatients] = useState();

  useEffect(() => {
    const fetchPatients = async () => {
      try {
        const response = await axios.get("/api/patients", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setPatients(response.data);
      } catch (error) {
        console.error("Error fetching patients:", error);
      }
    };
    fetchPatients();
  }, [token]);

  return (
    <div>
      <h2>Patients</h2>
      <ul>
        {patients.map((patient) => (
          <li key={patient.id}>
            <h3>{patient.name}</h3>
            {/* Add more patient details and actions here */}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default PatientList;