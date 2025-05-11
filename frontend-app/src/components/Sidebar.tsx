import React, { useContext } from 'react';
import { AppContext } from '../context/ChatContext';
// import { v4 as uuidv4 } from 'uuid'; // 이제 Context에서 ID를 생성하므로 여기서는 필요 없을 수 있습니다.

const Sidebar: React.FC = () => {
  const { state, dispatch } = useContext(AppContext)!;

  // 새 대화 생성
  const createNewConversation = () => {
    // const newId = uuidv4(); // ID는 이제 Context의 리듀서에서 생성합니다.
    const title = `Conversation ${state.conversations.length + 1}`; // 필요하다면 이 제목을 사용합니다.
    dispatch({
      type: 'CREATE_CONVERSATION', // ★★★ 'ADD_CONVERSATION'에서 변경!
      payload: { title }           // ★★★ ID는 보내지 않고, title만 보냅니다 (새 Context 정의에 따름).
                                   // 만약 새 Context의 CREATE_CONVERSATION이 payload를 받지 않는다면
                                   // payload 부분을 아예 생략할 수도 있습니다: dispatch({ type: 'CREATE_CONVERSATION' });
    });
  };

  // 대화 선택 핸들러
  const selectConversation = (id: string) => {
    dispatch({
      type: 'SELECT_CONVERSATION',      // ★★★ 'SET_CURRENT_CONVERSATION'에서 변경!
      payload: { conversationId: id } // ★★★ { id } 에서 { conversationId: id } 로 변경!
    });
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
            {/* 새 Context의 Conversation 타입에 title이 optional일 수 있으므로, 없을 경우 대비 */}
            {conv.title || `Conversation ${conv.id.slice(0, 8)}`}
          </li>
        ))}
      </ul>
    </div>
  );
};

export default Sidebar;