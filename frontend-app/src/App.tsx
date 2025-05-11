import React from 'react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import TaskDashboard from './components/TaskDashboard';

function App() {
  return (
    <div className="app-container">
      <Sidebar />
      <ChatWindow />
      <TaskDashboard />
    </div>
  );
}

export default App;
