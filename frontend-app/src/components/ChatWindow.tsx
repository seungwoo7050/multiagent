// export default ChatWindow;
import React, { useContext, useState, useEffect, useRef } from 'react'; // useEffect, useRef 추가
import axios from 'axios';
import { AppContext } from '../context/ChatContext';

const ChatWindow: React.FC = () => {
  const { state, dispatch } = useContext(AppContext)!;
  const currentConvFromState = state.conversations.find(c => c.id === state.currentConversationId);
  const [inputText, setInputText] = useState('');
  const [isSending, setIsSending] = useState(false); // 로딩 상태 추가

  // 현재 WebSocket 연결을 저장할 ref
  const socketRef = useRef<WebSocket | null>(null);

  // 컴포넌트 언마운트 시 또는 currentConv 변경 시 WebSocket 연결 정리
  useEffect(() => {
    return () => {
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        console.log('[ChatWindow Cleanup] Closing WebSocket connection for previous conversation.');
        socketRef.current.close();
        socketRef.current = null;
      }
    };
  }, [state.currentConversationId]); // currentConversationId를 의존성 배열에 추가



  const sendMessage = async () => {
    const currentConv = state.conversations.find(c => c.id === state.currentConversationId);

    if (!currentConv || inputText.trim() === '' || isSending) { // isSending 추가
      console.log('[sendMessage] Aborted: No current conversation, empty input, or already sending.');
      return;
    }

    const userMessage = inputText.trim();
    console.log('[sendMessage] User message:', userMessage, 'for conversationId:', currentConv.id);

    setIsSending(true); // 로딩 시작
    dispatch({
      type: 'ADD_MESSAGE',
      payload: { conversationId: currentConv.id, message: { sender: 'user', content: userMessage } }
    });
    setInputText('');

    const API_BASE_URL = 'http://localhost:8000';
    const API_PREFIX = '/api/v1'; // 실제 사용하는 API Prefix로 변경 (환경 변수 등으로 관리 권장)
    const graphConfigToUse = 'task_division_workflow'; // 실제 유효한 그래프 이름으로 변경 필요

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

      // 이전 WebSocket 연결이 있다면 정리
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        console.log('[sendMessage] Closing existing WebSocket before opening new one.');
        socketRef.current.close();
      }

      // 수정된 WebSocket URL 구성
      const wsBaseUrl = API_BASE_URL.replace(/^http/, 'ws');
      const wsUrl = `${wsBaseUrl}${API_PREFIX}/ws/status/${taskIdFromResponse}`;
      console.log('[sendMessage] Attempting to connect to WebSocket URL:', wsUrl);

      const socket = new WebSocket(wsUrl);
      socketRef.current = socket; // ref에 현재 소켓 저장

      const originalTaskIdForThisSocket = taskIdFromResponse;


      socket.onopen = () => {
        console.log('[WebSocket onopen] Connection successful for taskId:', originalTaskIdForThisSocket);
        setIsSending(false);
        const currentActiveConvId = state.currentConversationId; // ★★★ 현재 state에서 가져옴
        if (currentActiveConvId) { // ★★★ null 체크 추가
            dispatch({
                type: 'STATUS_UPDATE',
                payload: {
                  conversationId: currentActiveConvId, // ★★★ 변경
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

          // ★★★ 핵심 수정: originalTaskIdForThisSocket과 비교 ★★★
          if (msg.task_id && msg.task_id !== originalTaskIdForThisSocket) {
            console.warn(`[WebSocket onmessage] Received message for different taskId. Socket is for: ${originalTaskIdForThisSocket}, Message is for: ${msg.task_id}. Ignoring.`);
            return;
          }

          // dispatch 시 사용할 conversationId는 현재 활성화된 대화의 ID를 사용합니다.
          const conversationIdForDispatch = state.currentConversationId;
          if (!conversationIdForDispatch) {
            console.error('[WebSocket onmessage] CRITICAL: currentConversationId is null in state. Cannot determine target conversation for dispatch. Message:', msg);
            return;
          }
          // (선택적) 추가 로그: 어떤 conversationId로 dispatch 하는지 확인
          // console.log(`[WebSocket onmessage] Dispatching to conversationId: ${conversationIdForDispatch} for task: ${msg.task_id}`);


          const eventType = msg.event_type;
          switch (eventType) {
            case 'status_update':
              console.log('[WebSocket onmessage] Dispatching STATUS_UPDATE:', msg);
              dispatch({
                type: 'STATUS_UPDATE',
                payload: {
                  conversationId: conversationIdForDispatch, // ★★★ 변경
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
                  conversationId: conversationIdForDispatch, // ★★★ 변경
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
                  payload: { conversationId: conversationIdForDispatch, finalAnswer: msg.final_answer } // ★★★ 변경
                });
              }
              if (msg.error_message) {
                dispatch({
                  type: 'FINAL_RESULT',
                  payload: { conversationId: conversationIdForDispatch, errorMessage: msg.error_message } // ★★★ 변경
                });
              }
              console.log('[WebSocket onmessage] Closing socket after final_result for taskId:', originalTaskIdForThisSocket);
              socket.close(1000, "Task completed");
              socketRef.current = null;
              setIsSending(false);
              break;
            case 'error':
              console.log('[WebSocket onmessage] Dispatching ERROR from server message:', msg);
              dispatch({
                type: 'ERROR',
                payload: { conversationId: conversationIdForDispatch, errorMessage: msg.message } // ★★★ 변경
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
          const convIdForError = state.currentConversationId;
          if (convIdForError) {
            dispatch({
              type: 'ERROR',
              payload: { conversationId: convIdForError, errorMessage: '수신 데이터 처리 중 오류 발생' }
            });
          }
          setIsSending(false);
        }
      };


      socket.onerror = (errorEvent) => {
        console.error('[WebSocket onerror] WebSocket error occurred:', errorEvent, 'URL attempted:', wsUrl);
        setIsSending(false);
        const convIdForError = state.currentConversationId; // ★★★ 현재 state에서 가져옴
        if (convIdForError) { // ★★★ null 체크 추가
          dispatch({
            type: 'ERROR',
            payload: { conversationId: convIdForError, errorMessage: 'WebSocket 연결 중 심각한 오류 발생' } // ★★★ 변경
          });
        }
        socketRef.current = null;
      };


      socket.onclose = (closeEvent) => {
        console.log('[WebSocket onclose] Connection closed. Code:', closeEvent.code, 'Reason:', `"${closeEvent.reason}"`, 'WasClean:', closeEvent.wasClean);
        setIsSending(false); // 연결 종료 시 로딩 종료
        if (currentConv && !closeEvent.wasClean && socketRef.current) { // socketRef.current 체크 추가: 이미 닫힌 소켓에 대한 중복 처리 방지
           dispatch({
              type: 'STATUS_UPDATE',
              payload: {
                conversationId: currentConv.id,
                status: 'disconnected_ws',
                detail: `WebSocket 연결이 예기치 않게 닫힘 (코드: ${closeEvent.code})`
              }
            });
        }
        socketRef.current = null; // 연결 종료 시 ref 초기화
      };

    } catch (err: any) {
      console.error('[sendMessage] HTTP POST request failed:', err);
      setIsSending(false); // HTTP 에러 시 로딩 종료
      let errorMessage = '대화 시작 중 알 수 없는 오류 발생';
      // ... (기존 axios 에러 처리 로직) ...
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

      if (currentConv) {
        dispatch({
          type: 'ERROR',
          payload: { conversationId: currentConv.id, errorMessage: errorMessage }
        });
      }
    }
  };

  return (
    <div className="chat-window">
      <div className="messages">
        {currentConvFromState ? (
          <>
            {currentConvFromState.messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.sender}`}>
                <strong>{msg.sender === 'user' ? 'User' : 'Assistant'}:</strong> {msg.content}
              </div>
            ))}
            {/* 로딩 상태 또는 작업 상태 메시지 */}
            {(isSending || (currentConvFromState.status && !['completed', 'failed', 'error', 'disconnected_ws'].includes(currentConvFromState.status))) && (
              <div className="message assistant">
                <em>
                  {currentConvFromState.status === 'connected_ws' || currentConvFromState.status === 'pending_connection' ? '연결 중...' :
                   currentConvFromState.status === 'running' || currentConvFromState.status === 'status_update' ? `작업 진행 중 (${currentConvFromState.statusDetail || currentConvFromState.status})...` :
                   isSending ? '요청 전송 중...' :
                   `상태: ${currentConvFromState.status}...`}
                </em>
              </div>
            )}
            {/* 오류 발생 시 오류 메시지 표시 */}
            {(currentConvFromState.status === 'failed' || currentConvFromState.status === 'error' || currentConvFromState.status === 'disconnected_ws') && currentConvFromState.errorMessage && (
              <div className="message error">
                <strong>Error:</strong> {currentConvFromState.errorMessage}
              </div>
            )}
          </>
        ) : (
          <p>좌측에서 대화를 선택하거나 새로 시작하세요.</p>
        )}
      </div>
      <div className="input-area">
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onKeyPress={(e) => { if (e.key === 'Enter' && !isSending) sendMessage(); }} // 엔터키로 전송, isSending 중복 방지
          placeholder="메시지를 입력하세요..."
          disabled={isSending || !currentConvFromState} // 로딩 중이거나 대화 없을 시 비활성화
        />
        <button onClick={sendMessage} disabled={isSending || !currentConvFromState || inputText.trim() === ''}>
          {isSending ? '전송 중...' : '전송'}
        </button>
      </div>
    </div>
  );
};

export default ChatWindow;