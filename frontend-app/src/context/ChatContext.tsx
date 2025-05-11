// import React, { createContext, useReducer } from 'react';
// import type { ReactNode } from 'react'; 

// /** Message: 사용자 또는 Assistant의 채팅 메시지 */
// interface Message {
//   sender: 'user' | 'assistant';
//   content: string;
// }

// /** Conversation: 각 대화에 대한 데이터 구조 */
// interface Conversation {
//   id: string;
//   title: string;
//   messages: Message[];
//   taskId?: string;
//   status?: string;
//   statusDetail?: string;
//   currentNode?: string;
//   nextNode?: string;
//   intermediateResults?: { nodeId: string; resultStepName: string; data: any }[];
//   errorMessage?: string;
// }

// /** 전역 상태 구조 */
// interface AppState {
//   conversations: Conversation[];
//   currentConversationId: string | null;
// }

// /** 액션 타입 정의 */
// type Action =
//   | { type: 'ADD_CONVERSATION'; payload: { id: string; title: string } }
//   | { type: 'SET_CURRENT_CONVERSATION'; payload: { id: string } }
//   | { type: 'ADD_MESSAGE'; payload: { conversationId: string; message: Message } }
//   | { type: 'SET_TASK_ID'; payload: { conversationId: string; taskId: string } }
//   | { type: 'STATUS_UPDATE'; payload: { conversationId: string; status: string; detail?: string; currentNode?: string; nextNode?: string } }
//   | { type: 'ADD_INTERMEDIATE_RESULT'; payload: { conversationId: string; nodeId: string; resultStepName: string; data: any } }
//   | { type: 'FINAL_RESULT'; payload: { conversationId: string; finalAnswer?: string; errorMessage?: string } }
//   | { type: 'ERROR'; payload: { conversationId: string; errorMessage: string } };

// /** LocalStorage에서 초기 대화 목록 불러오기 (없으면 빈 배열) */
// const storedList = JSON.parse(localStorage.getItem('conversations') || '[]');
// const initialState: AppState = {
//   conversations: Array.isArray(storedList) ? storedList : [],
//   currentConversationId: Array.isArray(storedList) && storedList.length > 0 ? storedList[storedList.length - 1].id : null
// };

// /** 리듀서: 액션에 따라 상태를 변경합니다. */
// function appReducer(state: AppState, action: Action): AppState {
//   switch (action.type) {
//     case 'ADD_CONVERSATION': {
//       const { id, title } = action.payload;
//       const newConv: Conversation = { id, title, messages: [], intermediateResults: [] };
//       return {
//         ...state,
//         conversations: [...state.conversations, newConv],
//         currentConversationId: id  // 새 대화 시작 시 해당 대화를 활성화
//       };
//     }
//     case 'SET_CURRENT_CONVERSATION': {
//       return { ...state, currentConversationId: action.payload.id };
//     }
//     case 'ADD_MESSAGE': {
//       const { conversationId, message } = action.payload;
//       return {
//         ...state,
//         conversations: state.conversations.map(conv =>
//           conv.id === conversationId
//             ? { ...conv, messages: [...conv.messages, message] }
//             : conv
//         )
//       };
//     }
//     case 'SET_TASK_ID': {
//       const { conversationId, taskId } = action.payload;
//       return {
//         ...state,
//         conversations: state.conversations.map(conv =>
//           conv.id === conversationId
//             ? { ...conv, taskId, status: 'running' }
//             : conv
//         )
//       };
//     }
//     case 'STATUS_UPDATE': {
//       const { conversationId, status, detail, currentNode, nextNode } = action.payload;
//       return {
//         ...state,
//         conversations: state.conversations.map(conv =>
//           conv.id === conversationId
//             ? {
//                 ...conv,
//                 status,
//                 // detail, currentNode, nextNode는 있을 때만 업데이트
//                 statusDetail: detail ?? conv.statusDetail,
//                 currentNode: currentNode ?? conv.currentNode,
//                 nextNode: nextNode ?? conv.nextNode
//               }
//             : conv
//         )
//       };
//     }
//     case 'ADD_INTERMEDIATE_RESULT': {
//       const { conversationId, nodeId, resultStepName, data } = action.payload;
//       return {
//         ...state,
//         conversations: state.conversations.map(conv =>
//           conv.id === conversationId
//             ? {
//                 ...conv,
//                 intermediateResults: [
//                   ...(conv.intermediateResults || []),
//                   { nodeId, resultStepName, data }
//                 ]
//               }
//             : conv
//         )
//       };
//     }
//     case 'FINAL_RESULT': {
//       const { conversationId, finalAnswer, errorMessage } = action.payload;
//       return {
//         ...state,
//         conversations: state.conversations.map(conv => {
//           if (conv.id !== conversationId) return conv;
//           if (errorMessage) {
//             // 오류가 포함된 최종 결과
//             return { ...conv, status: 'failed', errorMessage };
//           }
//           if (finalAnswer) {
//             // 최종 답변이 도착하면 assistant 메시지 추가
//             return {
//               ...conv,
//               status: 'completed',
//               messages: [...conv.messages, { sender: 'assistant', content: finalAnswer }]
//             };
//           }
//           return conv;
//         })
//       };
//     }
//     case 'ERROR': {
//       const { conversationId, errorMessage } = action.payload;
//       return {
//         ...state,
//         conversations: state.conversations.map(conv =>
//           conv.id === conversationId
//             ? { ...conv, status: 'error', errorMessage }
//             : conv
//         )
//       };
//     }
//     default:
//       return state;
//   }
// }

// /** Context 생성: state와 dispatch를 제공 */
// interface AppContextProps {
//   state: AppState;
//   dispatch: React.Dispatch<Action>;
// }
// export const AppContext = createContext<AppContextProps | undefined>(undefined);

// /** Context Provider 컴포넌트 */
// export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
//   const [state, dispatch] = useReducer(appReducer, initialState);

//   // 상태 변화가 있을 때 localStorage에 저장하여 지속성 확보
//   React.useEffect(() => {
//     localStorage.setItem('conversations', JSON.stringify(state.conversations));
//   }, [state.conversations]);

//   return (
//     <AppContext.Provider value={{ state, dispatch }}>
//       {children}
//     </AppContext.Provider>
//   );
// };

import React, { createContext, useReducer } from 'react';
import type { ReactNode } from 'react'; 
import { v4 as uuidv4 } from 'uuid';

// Define types
type Message = {
  sender: 'user' | 'assistant';
  content: string;
};

type Conversation = {
  id: string;
  title?: string;
  messages: Message[];
  taskId?: string;
  status?: string;
  statusDetail?: string;
  errorMessage?: string;
  intermediateResults?: any[];
  currentNode?: string;
  nextNode?: string;
};

type AppState = {
  conversations: Conversation[];
  currentConversationId: string | null;
};

type Action =
  | { type: 'CREATE_CONVERSATION'; payload?: { title?: string } }
  | { type: 'SELECT_CONVERSATION'; payload: { conversationId: string } }
  | { type: 'DELETE_CONVERSATION'; payload: { conversationId: string } }
  | { type: 'ADD_MESSAGE'; payload: { conversationId: string; message: Message } }
  | { type: 'SET_TASK_ID'; payload: { conversationId: string; taskId: string } }
  | { type: 'STATUS_UPDATE'; payload: { conversationId: string; status: string; detail?: string; currentNode?: string; nextNode?: string } }
  | { type: 'ADD_INTERMEDIATE_RESULT'; payload: { conversationId: string; nodeId: string; resultStepName: string; data: any } }
  | { type: 'FINAL_RESULT'; payload: { conversationId: string; finalAnswer?: string; errorMessage?: string } }
  | { type: 'ERROR'; payload: { conversationId: string; errorMessage: string } };

type AppContextType = {
  state: AppState;
  dispatch: React.Dispatch<Action>;
};

// Create context
export const AppContext = createContext<AppContextType | null>(null);

// Initial state - load from localStorage if available
const getInitialState = (): AppState => {
  const savedState = localStorage.getItem('chatAppState');
  if (savedState) {
    try {
      return JSON.parse(savedState);
    } catch (e) {
      console.error('Failed to parse saved state:', e);
    }
  }
  
  // Default initial state with one empty conversation
  const newConversationId = uuidv4();
  return {
    conversations: [
      {
        id: newConversationId,
        messages: []
      }
    ],
    currentConversationId: newConversationId
  };
};

// Reducer function
const appReducer = (state: AppState, action: Action): AppState => {
  let newState: AppState;

  switch (action.type) {
    case 'CREATE_CONVERSATION': {
      const newConversationId = uuidv4();
      newState = {
        ...state,
        conversations: [
          ...state.conversations,
          {
            id: newConversationId,
            title: action.payload?.title,
            messages: []
          }
        ],
        currentConversationId: newConversationId
      };
      break;
    }

    case 'SELECT_CONVERSATION': {
      newState = {
        ...state,
        currentConversationId: action.payload.conversationId
      };
      break;
    }

    case 'DELETE_CONVERSATION': {
      const { conversationId } = action.payload;
      const remainingConversations = state.conversations.filter(conv => conv.id !== conversationId);
      
      // If we're deleting the current conversation, select another one
      let newCurrentId = state.currentConversationId;
      if (newCurrentId === conversationId) {
        // Pick the most recent conversation, or null if none left
        newCurrentId = remainingConversations.length > 0 ? remainingConversations[0].id : null;
      }
      
      // If we have no conversations left, create a new empty one
      if (remainingConversations.length === 0) {
        const newId = uuidv4();
        remainingConversations.push({
          id: newId,
          messages: []
        });
        newCurrentId = newId;
      }
      
      newState = {
        ...state,
        conversations: remainingConversations,
        currentConversationId: newCurrentId
      };
      break;
    }

    // Existing reducer cases...
    case 'ADD_MESSAGE': {
      newState = {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === action.payload.conversationId
            ? {
                ...conv,
                messages: [...conv.messages, action.payload.message]
              }
            : conv
        )
      };
      break;
    }

    case 'SET_TASK_ID': {
      newState = {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === action.payload.conversationId
            ? {
                ...conv,
                taskId: action.payload.taskId
              }
            : conv
        )
      };
      break;
    }

    case 'STATUS_UPDATE': {
      newState = {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === action.payload.conversationId
            ? {
                ...conv,
                status: action.payload.status,
                statusDetail: action.payload.detail,
                currentNode: action.payload.currentNode,
                nextNode: action.payload.nextNode
              }
            : conv
        )
      };
      break;
    }

    case 'ADD_INTERMEDIATE_RESULT': {
      newState = {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === action.payload.conversationId
            ? {
                ...conv,
                intermediateResults: [
                  ...(conv.intermediateResults || []),
                  {
                    nodeId: action.payload.nodeId,
                    resultStepName: action.payload.resultStepName,
                    data: action.payload.data
                  }
                ]
              }
            : conv
        )
      };
      break;
    }

    case 'FINAL_RESULT': {
      newState = {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === action.payload.conversationId
            ? {
                ...conv,
                status: 'completed',
                messages: action.payload.finalAnswer
                  ? [
                      ...conv.messages,
                      {
                        sender: 'assistant',
                        content: action.payload.finalAnswer
                      }
                    ]
                  : conv.messages,
                errorMessage: action.payload.errorMessage
              }
            : conv
        )
      };
      break;
    }

    case 'ERROR': {
      newState = {
        ...state,
        conversations: state.conversations.map(conv =>
          conv.id === action.payload.conversationId
            ? {
                ...conv,
                status: 'error',
                errorMessage: action.payload.errorMessage
              }
            : conv
        )
      };
      break;
    }

    default:
      return state;
  }

  // Save to localStorage after any state change
  localStorage.setItem('chatAppState', JSON.stringify(newState));
  return newState;
};

// Provider component
export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, getInitialState());

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
};