import React, { createContext, useReducer } from 'react';
import type { ReactNode } from 'react'; 

/** Message: 사용자 또는 Assistant의 채팅 메시지 */
interface Message {
  sender: 'user' | 'assistant';
  content: string;
}

/** Conversation: 각 대화에 대한 데이터 구조 */
interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  taskId?: string;
  status?: string;
  statusDetail?: string;
  currentNode?: string;
  nextNode?: string;
  intermediateResults?: { nodeId: string; resultStepName: string; data: any }[];
  errorMessage?: string;
}

/** 전역 상태 구조 */
interface AppState {
  conversations: Conversation[];
  currentConversationId: string | null;
}

/** 액션 타입 정의 */
type Action =
  | { type: 'ADD_CONVERSATION'; payload: { id: string; title: string } }
  | { type: 'SET_CURRENT_CONVERSATION'; payload: { id: string } }
  | { type: 'ADD_MESSAGE'; payload: { conversationId: string; message: Message } }
  | { type: 'SET_TASK_ID'; payload: { conversationId: string; taskId: string } }
  | { type: 'STATUS_UPDATE'; payload: { conversationId: string; status: string; detail?: string; currentNode?: string; nextNode?: string } }
  | { type: 'ADD_INTERMEDIATE_RESULT'; payload: { conversationId: string; nodeId: string; resultStepName: string; data: any } }
  | { type: 'FINAL_RESULT'; payload: { conversationId: string; finalAnswer?: string; errorMessage?: string } }
  | { type: 'ERROR'; payload: { conversationId: string; errorMessage: string } };

/** LocalStorage에서 초기 대화 목록 불러오기 (없으면 빈 배열) */
const storedList = JSON.parse(localStorage.getItem('conversations') || '[]');
const initialState: AppState = {
  conversations: Array.isArray(storedList) ? storedList : [],
  currentConversationId: Array.isArray(storedList) && storedList.length > 0 ? storedList[storedList.length - 1].id : null
};

/** 리듀서: 액션에 따라 상태를 변경합니다. */
function appReducer(state: AppState, action: Action): AppState {
  switch (action.type) {
    case 'ADD_CONVERSATION': {
      const { id, title } = action.payload;
      const newConv: Conversation = { id, title, messages: [], intermediateResults: [] };
      return {
        ...state,
        conversations: [...state.conversations, newConv],
        currentConversationId: id  // 새 대화 시작 시 해당 대화를 활성화
      };
    }
    case 'SET_CURRENT_CONVERSATION': {
      return { ...state, currentConversationId: action.payload.id };
    }
    case 'ADD_MESSAGE': {
      const { conversationId, message } = action.payload;
      return {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === conversationId
            ? { ...conv, messages: [...conv.messages, message] }
            : conv
        )
      };
    }
    case 'SET_TASK_ID': {
      const { conversationId, taskId } = action.payload;
      return {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === conversationId
            ? { ...conv, taskId, status: 'running' }
            : conv
        )
      };
    }
    case 'STATUS_UPDATE': {
      const { conversationId, status, detail, currentNode, nextNode } = action.payload;
      return {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === conversationId
            ? {
                ...conv,
                status,
                // detail, currentNode, nextNode는 있을 때만 업데이트
                statusDetail: detail ?? conv.statusDetail,
                currentNode: currentNode ?? conv.currentNode,
                nextNode: nextNode ?? conv.nextNode
              }
            : conv
        )
      };
    }
    case 'ADD_INTERMEDIATE_RESULT': {
      const { conversationId, nodeId, resultStepName, data } = action.payload;
      return {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === conversationId
            ? {
                ...conv,
                intermediateResults: [
                  ...(conv.intermediateResults || []),
                  { nodeId, resultStepName, data }
                ]
              }
            : conv
        )
      };
    }
    case 'FINAL_RESULT': {
      const { conversationId, finalAnswer, errorMessage } = action.payload;
      return {
        ...state,
        conversations: state.conversations.map(conv => {
          if (conv.id !== conversationId) return conv;
          if (errorMessage) {
            // 오류가 포함된 최종 결과
            return { ...conv, status: 'failed', errorMessage };
          }
          if (finalAnswer) {
            // 최종 답변이 도착하면 assistant 메시지 추가
            return {
              ...conv,
              status: 'completed',
              messages: [...conv.messages, { sender: 'assistant', content: finalAnswer }]
            };
          }
          return conv;
        })
      };
    }
    case 'ERROR': {
      const { conversationId, errorMessage } = action.payload;
      return {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === conversationId
            ? { ...conv, status: 'error', errorMessage }
            : conv
        )
      };
    }
    default:
      return state;
  }
}

/** Context 생성: state와 dispatch를 제공 */
interface AppContextProps {
  state: AppState;
  dispatch: React.Dispatch<Action>;
}
export const AppContext = createContext<AppContextProps | undefined>(undefined);

/** Context Provider 컴포넌트 */
export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // 상태 변화가 있을 때 localStorage에 저장하여 지속성 확보
  React.useEffect(() => {
    localStorage.setItem('conversations', JSON.stringify(state.conversations));
  }, [state.conversations]);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};
