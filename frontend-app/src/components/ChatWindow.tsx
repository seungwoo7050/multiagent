// // First, install the react-markdown package:
// // npm install react-markdown

// import React, { useContext, useState, useEffect, useRef } from 'react';
// import axios from 'axios';
// import { AppContext } from '../context/ChatContext';
// import ReactMarkdown from 'react-markdown'; // Import the Markdown renderer
// import './markdown-styles.css'; // Import the markdown styles

// const ChatWindow: React.FC = () => {
//   const { state, dispatch } = useContext(AppContext)!;
//   const currentConvFromState = state.conversations.find(c => c.id === state.currentConversationId);
//   const [inputText, setInputText] = useState('');
//   const [isSending, setIsSending] = useState(false);

//   // Current WebSocket connection reference
//   const socketRef = useRef<WebSocket | null>(null);

//   // Clean up WebSocket connection on unmount or conversation change
//   useEffect(() => {
//     return () => {
//       if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
//         console.log('[ChatWindow Cleanup] Closing WebSocket connection for previous conversation.');
//         socketRef.current.close();
//         socketRef.current = null;
//       }
//     };
//   }, [state.currentConversationId]);

//   const sendMessage = async () => {
//     const currentConv = state.conversations.find(c => c.id === state.currentConversationId);

//     if (!currentConv || inputText.trim() === '' || isSending) {
//       console.log('[sendMessage] Aborted: No current conversation, empty input, or already sending.');
//       return;
//     }

//     const userMessage = inputText.trim();
//     console.log('[sendMessage] User message:', userMessage, 'for conversationId:', currentConv.id);

//     setIsSending(true);
//     dispatch({
//       type: 'ADD_MESSAGE',
//       payload: { conversationId: currentConv.id, message: { sender: 'user', content: userMessage } }
//     });
//     setInputText('');

//     const API_BASE_URL = 'http://localhost:8000';
//     const API_PREFIX = '/api/v1';
//     const graphConfigToUse = 'task_division_workflow';

//     try {
//       console.log('[sendMessage] Attempting HTTP POST to /api/v1/run with payload:', {
//         original_input: userMessage,
//         graph_config_name: graphConfigToUse,
//         initial_metadata: { conversation_id: currentConv.id }
//       });

//       const response = await axios.post(`${API_BASE_URL}${API_PREFIX}/run`, {
//         original_input: userMessage,
//         graph_config_name: graphConfigToUse,
//         initial_metadata: { conversation_id: currentConv.id }
//       });

//       const taskIdFromResponse: string = response.data.task_id; 
//       console.log('[sendMessage] HTTP POST success. Received taskId:', taskIdFromResponse);

//       dispatch({ type: 'SET_TASK_ID', payload: { conversationId: currentConv.id, taskId: taskIdFromResponse } });

//       // 이전 WebSocket 연결이 있다면 정리
//       if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
//         console.log('[sendMessage] Closing existing WebSocket before opening new one.');
//         socketRef.current.close();
//       }

//       // WebSocket URL 구성
//       const wsBaseUrl = API_BASE_URL.replace(/^http/, 'ws');
//       const wsUrl = `${wsBaseUrl}${API_PREFIX}/ws/status/${taskIdFromResponse}`;
//       console.log('[sendMessage] Attempting to connect to WebSocket URL:', wsUrl);

//       const socket = new WebSocket(wsUrl);
//       socketRef.current = socket;

//       const originalTaskIdForThisSocket = taskIdFromResponse;

//       socket.onopen = () => {
//         console.log('[WebSocket onopen] Connection successful for taskId:', originalTaskIdForThisSocket);
//         setIsSending(false);
//         const currentActiveConvId = state.currentConversationId;
//         if (currentActiveConvId) {
//             dispatch({
//                 type: 'STATUS_UPDATE',
//                 payload: {
//                   conversationId: currentActiveConvId,
//                   status: 'connected_ws',
//                   detail: `WebSocket 연결됨 (Task: ${originalTaskIdForThisSocket})`
//                 }
//             });
//         } else {
//             console.warn("[WebSocket onopen] No current conversation ID found in state. Cannot dispatch STATUS_UPDATE for onopen.");
//         }
//       };

//       socket.onmessage = (event) => {
//         console.log('[WebSocket onmessage] Raw data received:', event.data);
//         try {
//           const msg = JSON.parse(event.data as string);
//           console.log('[WebSocket onmessage] Parsed message:', msg);

//           if (msg.task_id && msg.task_id !== originalTaskIdForThisSocket) {
//             console.warn(`[WebSocket onmessage] Received message for different taskId. Socket is for: ${originalTaskIdForThisSocket}, Message is for: ${msg.task_id}. Ignoring.`);
//             return;
//           }

//           const conversationIdForDispatch = state.currentConversationId;
//           if (!conversationIdForDispatch) {
//             console.error('[WebSocket onmessage] CRITICAL: currentConversationId is null in state. Cannot determine target conversation for dispatch. Message:', msg);
//             return;
//           }

//           const eventType = msg.event_type;
//           switch (eventType) {
//             case 'status_update':
//               console.log('[WebSocket onmessage] Dispatching STATUS_UPDATE:', msg);
//               dispatch({
//                 type: 'STATUS_UPDATE',
//                 payload: {
//                   conversationId: conversationIdForDispatch,
//                   status: msg.status,
//                   detail: msg.detail,
//                   currentNode: msg.current_node,
//                   nextNode: msg.next_node
//                 }
//               });
//               break;
//             case 'intermediate_result':
//               console.log('[WebSocket onmessage] Dispatching ADD_INTERMEDIATE_RESULT:', msg);
//               dispatch({
//                 type: 'ADD_INTERMEDIATE_RESULT',
//                 payload: {
//                   conversationId: conversationIdForDispatch,
//                   nodeId: msg.node_id,
//                   resultStepName: msg.result_step_name,
//                   data: msg.data
//                 }
//               });
//               break;
//             case 'final_result':
//               console.log('[WebSocket onmessage] Dispatching FINAL_RESULT:', msg);
//               if (msg.final_answer !== undefined && msg.final_answer !== null) {
//                 dispatch({
//                   type: 'FINAL_RESULT',
//                   payload: { conversationId: conversationIdForDispatch, finalAnswer: msg.final_answer }
//                 });
//               }
//               if (msg.error_message) {
//                 dispatch({
//                   type: 'FINAL_RESULT',
//                   payload: { conversationId: conversationIdForDispatch, errorMessage: msg.error_message }
//                 });
//               }
//               console.log('[WebSocket onmessage] Closing socket after final_result for taskId:', originalTaskIdForThisSocket);
//               socket.close(1000, "Task completed");
//               socketRef.current = null;
//               setIsSending(false);
//               break;
//             case 'error':
//               console.log('[WebSocket onmessage] Dispatching ERROR from server message:', msg);
//               dispatch({
//                 type: 'ERROR',
//                 payload: { conversationId: conversationIdForDispatch, errorMessage: msg.message }
//               });
//               console.log('[WebSocket onmessage] Closing socket after server error message for taskId:', originalTaskIdForThisSocket);
//               socket.close(1000, "Server error reported");
//               socketRef.current = null;
//               setIsSending(false);
//               break;
//             default:
//               console.warn('[WebSocket onmessage] Unknown event_type:', eventType, 'Full message:', msg);
//           }
//         } catch (e) {
//           console.error('[WebSocket onmessage] Error parsing JSON or dispatching:', e, 'Original data:', event.data);
//           const convIdForError = state.currentConversationId;
//           if (convIdForError) {
//             dispatch({
//               type: 'ERROR',
//               payload: { conversationId: convIdForError, errorMessage: '수신 데이터 처리 중 오류 발생' }
//             });
//           }
//           setIsSending(false);
//         }
//       };

//       socket.onerror = (errorEvent) => {
//         console.error('[WebSocket onerror] WebSocket error occurred:', errorEvent, 'URL attempted:', wsUrl);
//         setIsSending(false);
//         const convIdForError = state.currentConversationId;
//         if (convIdForError) {
//           dispatch({
//             type: 'ERROR',
//             payload: { conversationId: convIdForError, errorMessage: 'WebSocket 연결 중 심각한 오류 발생' }
//           });
//         }
//         socketRef.current = null;
//       };

//       socket.onclose = (closeEvent) => {
//         console.log('[WebSocket onclose] Connection closed. Code:', closeEvent.code, 'Reason:', `"${closeEvent.reason}"`, 'WasClean:', closeEvent.wasClean);
//         setIsSending(false);
//         if (currentConv && !closeEvent.wasClean && socketRef.current) {
//            dispatch({
//               type: 'STATUS_UPDATE',
//               payload: {
//                 conversationId: currentConv.id,
//                 status: 'disconnected_ws',
//                 detail: `WebSocket 연결이 예기치 않게 닫힘 (코드: ${closeEvent.code})`
//               }
//             });
//         }
//         socketRef.current = null;
//       };

//     } catch (err: any) {
//       console.error('[sendMessage] HTTP POST request failed:', err);
//       setIsSending(false);
//       let errorMessage = '대화 시작 중 알 수 없는 오류 발생';
      
//       if (axios.isAxiosError(err)) {
//         if (err.response) {
//           console.error('Error data (HTTP POST):', err.response.data);
//           console.error('Error status (HTTP POST):', err.response.status);
//           const detail = err.response.data?.detail;
//           if (Array.isArray(detail) && detail.length > 0 && detail[0].msg) {
//               errorMessage = `요청 오류: ${detail[0].msg} (필드: ${detail[0].loc.join('.')})`;
//           } else if (typeof detail === 'string') {
//               errorMessage = `요청 오류: ${detail}`;
//           } else {
//               errorMessage = `서버 응답 오류 (상태 코드: ${err.response.status})`;
//           }
//         } else if (err.request) {
//           errorMessage = '서버에서 응답이 없습니다. 네트워크 또는 백엔드 서버 상태를 확인하세요.';
//         } else {
//           errorMessage = `요청 설정 오류: ${err.message}`;
//         }
//       }

//       if (currentConv) {
//         dispatch({
//           type: 'ERROR',
//           payload: { conversationId: currentConv.id, errorMessage: errorMessage }
//         });
//       }
//     }
//   };

//   return (
//     <div className="chat-window">
//       <div className="messages">
//         {currentConvFromState ? (
//           <>
//             {currentConvFromState.messages.map((msg, idx) => (
//               <div key={idx} className={`message ${msg.sender}`}>
//                 <strong>{msg.sender === 'user' ? 'User' : 'Assistant'}:</strong>
//                 {/* Replace the regular content display with ReactMarkdown */}
//                 {msg.sender === 'assistant' ? (
//                   <div className="markdown-content">
//                     <ReactMarkdown>{msg.content}</ReactMarkdown>
//                   </div>
//                 ) : (
//                   <span>{msg.content}</span> // Keep user messages as plain text
//                 )}
//               </div>
//             ))}
//             {/* Loading state or task status messages */}
//             {(isSending || (currentConvFromState.status && !['completed', 'failed', 'error', 'disconnected_ws'].includes(currentConvFromState.status))) && (
//               <div className="message assistant">
//                 <em>
//                   {currentConvFromState.status === 'connected_ws' || currentConvFromState.status === 'pending_connection' ? '연결 중...' :
//                    currentConvFromState.status === 'running' || currentConvFromState.status === 'status_update' ? `작업 진행 중 (${currentConvFromState.statusDetail || currentConvFromState.status})...` :
//                    isSending ? '요청 전송 중...' :
//                    `상태: ${currentConvFromState.status}...`}
//                 </em>
//               </div>
//             )}
//             {/* Error messages */}
//             {(currentConvFromState.status === 'failed' || currentConvFromState.status === 'error' || currentConvFromState.status === 'disconnected_ws') && currentConvFromState.errorMessage && (
//               <div className="message error">
//                 <strong>Error:</strong> {currentConvFromState.errorMessage}
//               </div>
//             )}
//           </>
//         ) : (
//           <p>좌측에서 대화를 선택하거나 새로 시작하세요.</p>
//         )}
//       </div>
//       <div className="input-area">
//         <input
//           type="text"
//           value={inputText}
//           onChange={(e) => setInputText(e.target.value)}
//           onKeyPress={(e) => { if (e.key === 'Enter' && !isSending) sendMessage(); }}
//           placeholder="메시지를 입력하세요..."
//           disabled={isSending || !currentConvFromState}
//         />
//         <button onClick={sendMessage} disabled={isSending || !currentConvFromState || inputText.trim() === ''}>
//           {isSending ? '전송 중...' : '전송'}
//         </button>
//       </div>
//     </div>
//   );
// };

// export default ChatWindow;

import React, { useContext, useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { AppContext } from '../context/ChatContext';
import ReactMarkdown from 'react-markdown'; // Import the Markdown renderer
import './markdown-styles.css'; // Import the markdown styles

const ChatWindow: React.FC = () => {
  const { state, dispatch } = useContext(AppContext)!;
  const currentConvFromState = state.conversations.find(c => c.id === state.currentConversationId);
  const [inputText, setInputText] = useState('');
  const [isSending, setIsSending] = useState(false); // 기존 isSending 상태 (boolean 유지)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false); // << 새로 추가된 상태

  // Current WebSocket connection reference
  const socketRef = useRef<WebSocket | null>(null);

  // Clean up WebSocket connection on unmount or conversation change
  useEffect(() => {
    return () => {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        console.log('[ChatWindow Cleanup] Closing WebSocket connection for previous conversation.');
        socketRef.current.close();
        socketRef.current = null;
      }
    };
  }, [state.currentConversationId]);

  // << 새로 추가된 함수: 대화 삭제 로직 >>
  const deleteConversation = () => {
    if (!currentConvFromState) return;

    // First close any active WebSocket connection
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      console.log('[deleteConversation] Closing WebSocket before deleting conversation');
      socketRef.current.close();
      socketRef.current = null;
    }

    // Dispatch delete action to context
    dispatch({
      type: 'DELETE_CONVERSATION', // ChatContext의 reducer에 이 액션 타입 처리가 필요합니다.
      payload: { conversationId: currentConvFromState.id }
    });

    setShowDeleteConfirm(false); // 확인 UI 숨기기

    console.log(`[deleteConversation] Conversation ${currentConvFromState.id} deleted`);
    // 참고: currentConversationId를 null로 설정하거나 다른 대화로 변경하는 로직이
    // DELETE_CONVERSATION 리듀서 또는 이 함수 내에 추가로 필요할 수 있습니다.
  };

  // << 새로 추가된 함수: 삭제 취소 로직 >>
  const cancelDelete = () => {
    setShowDeleteConfirm(false);
  };

  const sendMessage = async () => {
    const currentConv = state.conversations.find(c => c.id === state.currentConversationId);

    if (!currentConv || inputText.trim() === '' || isSending) {
      console.log('[sendMessage] Aborted: No current conversation, empty input, or already sending.');
      return;
    }

    const userMessage = inputText.trim();
    console.log('[sendMessage] User message:', userMessage, 'for conversationId:', currentConv.id);

    setIsSending(true);
    dispatch({
      type: 'ADD_MESSAGE',
      payload: { conversationId: currentConv.id, message: { sender: 'user', content: userMessage } }
    });
    setInputText('');

    const API_BASE_URL = 'http://localhost:8000';
    const API_PREFIX = '/api/v1';
    const graphConfigToUse = 'task_division_workflow';

    try {
      console.log('[sendMessage] Attempting HTTP POST to /api/v1/run with payload:', {
        original_input: userMessage,
        graph_config_name: graphConfigToUse,
        initial_metadata: { conversation_id: currentConv.id }
      });

      const response = await axios.post(`${API_BASE_URL}${API_PREFIX}/run`, {
        original_input: userMessage,
        graph_config_name: graphConfigToUse,
        initial_metadata: { conversation_id: currentConv.id }
      });

      const taskIdFromResponse: string = response.data.task_id;
      console.log('[sendMessage] HTTP POST success. Received taskId:', taskIdFromResponse);

      dispatch({ type: 'SET_TASK_ID', payload: { conversationId: currentConv.id, taskId: taskIdFromResponse } });

      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        console.log('[sendMessage] Closing existing WebSocket before opening new one.');
        socketRef.current.close();
      }

      const wsBaseUrl = API_BASE_URL.replace(/^http/, 'ws');
      const wsUrl = `${wsBaseUrl}${API_PREFIX}/ws/status/${taskIdFromResponse}`;
      console.log('[sendMessage] Attempting to connect to WebSocket URL:', wsUrl);

      const socket = new WebSocket(wsUrl);
      socketRef.current = socket;

      const originalTaskIdForThisSocket = taskIdFromResponse;

      socket.onopen = () => {
        console.log('[WebSocket onopen] Connection successful for taskId:', originalTaskIdForThisSocket);
        setIsSending(false); // 여기서 isSending을 false로 설정
        const currentActiveConvId = state.currentConversationId; // state에서 최신 ID 가져오기
        if (currentActiveConvId) { // currentActiveConvId가 유효한지 확인
            dispatch({
                type: 'STATUS_UPDATE',
                payload: {
                  conversationId: currentActiveConvId,
                  status: 'connected_ws',
                  detail: `WebSocket 연결됨 (Task: ${originalTaskIdForThisSocket})`
                }
            });
        } else {
            console.warn("[WebSocket onopen] No current conversation ID found in state. Cannot dispatch STATUS_UPDATE for onopen.");
        }
      };

      socket.onmessage = (event) => {
        console.log('[WebSocket onmessage] Raw data received:', event.data);
        try {
          const msg = JSON.parse(event.data as string);
          console.log('[WebSocket onmessage] Parsed message:', msg);

          if (msg.task_id && msg.task_id !== originalTaskIdForThisSocket) {
            console.warn(`[WebSocket onmessage] Received message for different taskId. Socket is for: ${originalTaskIdForThisSocket}, Message is for: ${msg.task_id}. Ignoring.`);
            return;
          }

          // 중요: 디스패치 시점의 state.currentConversationId를 사용해야 합니다.
          // 클로저 문제로 인해 sendMessage 함수 스코프의 currentConv.id를 사용하면
          // 대화가 변경된 후에도 이전 대화 ID로 디스패치될 수 있습니다.
          const conversationIdForDispatch = state.currentConversationId;
          if (!conversationIdForDispatch) {
            console.error('[WebSocket onmessage] CRITICAL: currentConversationId is null in state. Cannot determine target conversation for dispatch. Message:', msg);
            return;
          }

          const eventType = msg.event_type;
          switch (eventType) {
            case 'status_update':
              console.log('[WebSocket onmessage] Dispatching STATUS_UPDATE:', msg);
              dispatch({
                type: 'STATUS_UPDATE',
                payload: {
                  conversationId: conversationIdForDispatch,
                  status: msg.status,
                  detail: msg.detail,
                  currentNode: msg.current_node,
                  nextNode: msg.next_node
                }
              });
              break;
            case 'intermediate_result':
              console.log('[WebSocket onmessage] Dispatching ADD_INTERMEDIATE_RESULT:', msg);
              dispatch({
                type: 'ADD_INTERMEDIATE_RESULT',
                payload: {
                  conversationId: conversationIdForDispatch,
                  nodeId: msg.node_id,
                  resultStepName: msg.result_step_name,
                  data: msg.data
                }
              });
              break;
            case 'final_result':
              console.log('[WebSocket onmessage] Dispatching FINAL_RESULT:', msg);
              if (msg.final_answer !== undefined && msg.final_answer !== null) {
                dispatch({
                  type: 'FINAL_RESULT',
                  payload: { conversationId: conversationIdForDispatch, finalAnswer: msg.final_answer }
                });
              }
              if (msg.error_message) { // 에러 메시지도 final_result의 일부로 올 수 있음
                dispatch({
                  type: 'FINAL_RESULT', // 또는 별도의 ERROR 액션
                  payload: { conversationId: conversationIdForDispatch, errorMessage: msg.error_message }
                });
              }
              console.log('[WebSocket onmessage] Closing socket after final_result for taskId:', originalTaskIdForThisSocket);
              socket.close(1000, "Task completed");
              socketRef.current = null;
              setIsSending(false);
              break;
            case 'error': // 서버에서 명시적으로 'error' event_type으로 보내는 경우
              console.log('[WebSocket onmessage] Dispatching ERROR from server message:', msg);
              dispatch({
                type: 'ERROR',
                payload: { conversationId: conversationIdForDispatch, errorMessage: msg.message }
              });
              console.log('[WebSocket onmessage] Closing socket after server error message for taskId:', originalTaskIdForThisSocket);
              socket.close(1000, "Server error reported");
              socketRef.current = null;
              setIsSending(false);
              break;
            default:
              console.warn('[WebSocket onmessage] Unknown event_type:', eventType, 'Full message:', msg);
          }
        } catch (e) {
          console.error('[WebSocket onmessage] Error parsing JSON or dispatching:', e, 'Original data:', event.data);
          const convIdForError = state.currentConversationId; // 최신 ID 사용
          if (convIdForError) {
            dispatch({
              type: 'ERROR',
              payload: { conversationId: convIdForError, errorMessage: '수신 데이터 처리 중 오류 발생' }
            });
          }
          // isSending을 여기서도 false로 처리할지 여부는 상황에 따라 결정 (예: 재시도 로직이 없다면 false)
          // 만약 이미 final_result나 error로 인해 소켓이 닫히고 isSending이 false가 된 후라면 중복 호출될 수 있음
          // 하지만 안전하게 여기서도 false로 해두는 것이 좋을 수 있습니다.
          // setIsSending(false); // 이미 final_result, error에서 처리하므로 여기선 필요 없을 수 있음
        }
      };

      socket.onerror = (errorEvent) => {
        console.error('[WebSocket onerror] WebSocket error occurred:', errorEvent, 'URL attempted:', wsUrl);
        setIsSending(false);
        const convIdForError = state.currentConversationId; // 최신 ID 사용
        if (convIdForError) {
          dispatch({
            type: 'ERROR',
            payload: { conversationId: convIdForError, errorMessage: 'WebSocket 연결 중 심각한 오류 발생' }
          });
        }
        // socketRef.current = null; // onclose에서 처리되므로 중복일 수 있으나, 방어적으로 추가 가능
      };

      socket.onclose = (closeEvent) => {
        console.log('[WebSocket onclose] Connection closed. Code:', closeEvent.code, 'Reason:', `"${closeEvent.reason}"`, 'WasClean:', closeEvent.wasClean);
        setIsSending(false); // 연결 종료 시 isSending은 false
        // currentConv가 아닌 state.currentConversationId를 기준으로 해당 대화 찾기
        const convForCloseEvent = state.conversations.find(c => c.id === state.currentConversationId);

        // 작업이 완료되지 않은 상태 (wasClean=false)이고, 현재 참조하는 소켓과 동일할 때만 상태 업데이트
        // 이렇게 하면 이미 다른 작업으로 소켓이 교체된 후 이전 소켓의 onclose가 호출되어 상태를 잘못 업데이트하는 것을 방지.
        if (convForCloseEvent && !closeEvent.wasClean && socketRef.current === socket) {
           dispatch({
              type: 'STATUS_UPDATE',
              payload: {
                conversationId: convForCloseEvent.id, // 여기서도 최신 대화 ID 사용
                status: 'disconnected_ws',
                detail: `WebSocket 연결이 예기치 않게 닫힘 (코드: ${closeEvent.code})`
              }
            });
        }
        // 현재 참조된 소켓이 이 onclose 이벤트의 주체인 경우에만 null로 설정
        if (socketRef.current === socket) {
            socketRef.current = null;
        }
      };

    } catch (err: any) {
      console.error('[sendMessage] HTTP POST request failed:', err);
      setIsSending(false);
      let errorMessage = '대화 시작 중 알 수 없는 오류 발생';

      if (axios.isAxiosError(err)) {
        if (err.response) {
          console.error('Error data (HTTP POST):', err.response.data);
          console.error('Error status (HTTP POST):', err.response.status);
          const detail = err.response.data?.detail;
          if (Array.isArray(detail) && detail.length > 0 && detail[0].msg) {
              errorMessage = `요청 오류: ${detail[0].msg} (필드: ${detail[0].loc.join('.')})`;
          } else if (typeof detail === 'string') {
              errorMessage = `요청 오류: ${detail}`;
          } else {
              errorMessage = `서버 응답 오류 (상태 코드: ${err.response.status})`;
          }
        } else if (err.request) {
          errorMessage = '서버에서 응답이 없습니다. 네트워크 또는 백엔드 서버 상태를 확인하세요.';
        } else {
          errorMessage = `요청 설정 오류: ${err.message}`;
        }
      }

      // 에러 발생 시점의 currentConv (sendMessage 함수 스코프) 대신
      // state.currentConversationId를 사용해야 할 수도 있습니다.
      // 만약 HTTP 요청 중에 사용자가 대화를 빠르게 변경했다면, 에러 메시지가 이전 대화에 표시될 수 있습니다.
      // 여기서는 HTTP 요청 시작 시점의 대화에 에러를 연결하는 것이 일반적일 수 있습니다.
      // 하지만 웹소켓과 일관성을 위해 state.currentConversationId를 사용할 수도 있습니다.
      const conversationIdForError = currentConv ? currentConv.id : state.currentConversationId;

      if (conversationIdForError) {
        dispatch({
          type: 'ERROR',
          payload: { conversationId: conversationIdForError, errorMessage: errorMessage }
        });
      }
    }
  };

  return (
    <div className="chat-window">
      {/* << 새로 추가된 JSX: 채팅 헤더 및 삭제 버튼 >> */}
      {currentConvFromState && (
        <div className="chat-header">
          {/* 대화 제목을 표시합니다. 없으면 ID의 일부를 표시합니다. */}
          <h3>{currentConvFromState.title || `Conversation ${currentConvFromState.id.slice(0, 8)}`}</h3>
          {showDeleteConfirm ? (
            <div className="delete-confirm">
              <span>이 대화를 삭제하시겠습니까?</span>
              <button onClick={deleteConversation} className="confirm-btn">예</button>
              <button onClick={cancelDelete} className="cancel-btn">아니오</button>
            </div>
          ) : (
            <button
              onClick={() => setShowDeleteConfirm(true)}
              className="delete-btn"
              title="대화 삭제"
              disabled={isSending} // 메시지 전송 중에는 삭제 버튼 비활성화
            >
              🗑️ {/* 휴지통 아이콘 */}
            </button>
          )}
        </div>
      )}

      <div className="messages">
        {currentConvFromState ? (
          <>
            {currentConvFromState.messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.sender}`}>
                <strong>{msg.sender === 'user' ? 'User' : 'Assistant'}:</strong>
                {msg.sender === 'assistant' ? (
                  <div className="markdown-content">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                ) : (
                  <span>{msg.content}</span>
                )}
              </div>
            ))}
            {(isSending || (currentConvFromState.status && !['completed', 'failed', 'error', 'disconnected_ws'].includes(currentConvFromState.status))) && (
              <div className="message assistant">
                <em>
                  {currentConvFromState.status === 'connected_ws' || currentConvFromState.status === 'pending_connection' ? '연결 중...' :
                   currentConvFromState.status === 'running' || currentConvFromState.status === 'status_update' ? `작업 진행 중 (${currentConvFromState.statusDetail || currentConvFromState.status})...` :
                   isSending ? '요청 전송 중...' : // isSending이 true일 때 "요청 전송 중..." 표시
                   `상태: ${currentConvFromState.status}...`}
                </em>
              </div>
            )}
            {(currentConvFromState.status === 'failed' || currentConvFromState.status === 'error' || currentConvFromState.status === 'disconnected_ws') && currentConvFromState.errorMessage && (
              <div className="message error">
                <strong>Error:</strong> {currentConvFromState.errorMessage}
              </div>
            )}
          </>
        ) : (
          // << 메시지 변경: 기존 "좌측에서 대화를 선택하거나 새로 시작하세요." 에서 사용자 제공 코드로 변경 >>
          <p>Select a conversation from the left menu or start a new one.</p>
        )}
      </div>
      <div className="input-area">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={(e) => { if (e.key === 'Enter' && !isSending) sendMessage(); }}
          // << placeholder 메시지 변경 >>
          placeholder="Type your message..."
          disabled={isSending || !currentConvFromState}
        />
        <button onClick={sendMessage} disabled={isSending || !currentConvFromState || inputText.trim() === ''}>
          {/* << 버튼 텍스트 변경 >> */}
          {isSending ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
};

export default ChatWindow;