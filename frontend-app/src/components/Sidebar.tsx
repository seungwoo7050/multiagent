import React, { useContext } from 'react';
import { AppContext } from '../context/ChatContext';
import { v4 as uuidv4 } from 'uuid';

const Sidebar: React.FC = () => {
  const { state, dispatch } = useContext(AppContext)!;

  // 새 대화 생성
  const createNewConversation = () => {
    const newId = uuidv4();  // 고유 UUID 생성
    const title = `Conversation ${state.conversations.length + 1}`;
    dispatch({ type: 'ADD_CONVERSATION', payload: { id: newId, title } });
  };

  // 대화 선택 핸들러
  const selectConversation = (id: string) => {
    dispatch({ type: 'SET_CURRENT_CONVERSATION', payload: { id } });
  };

  return (
    <div className="sidebar">
      <button onClick={createNewConversation}>새 대화 시작</button>
      <ul>
        {state.conversations.map(conv => (
          <li 
            key={conv.id} 
            onClick={() => selectConversation(conv.id)}
            className={conv.id === state.currentConversationId ? 'active' : ''}
          >
            {conv.title}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Sidebar;
