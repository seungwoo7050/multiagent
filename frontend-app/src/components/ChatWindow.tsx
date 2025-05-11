// export default ChatWindow;
import React, { useContext, useState, useEffect, useRef } from 'react'; // useEffect, useRef 추가
import axios from 'axios';
import { AppContext } from '../context/ChatContext';

const ChatWindow: React.FC = () => {
  const { state, dispatch } = useContext(AppContext)!;
  const currentConv = state.conversations.find(c => c.id === state.currentConversationId);
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
  }, [currentConv?.id]); // currentConv.id가 바뀔 때마다 이전 소켓 정리


  const sendMessage = async () => {
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
    const graphConfigToUse = 'task_division_workflow'; // 실제 유효한 그래프 이름으로 변경 필요

    try {
      console.log('[sendMessage] Attempting HTTP POST to /api/v1/run with payload:', {
        original_input: userMessage,
        graph_config_name: graphConfigToUse,
        initial_metadata: { conversation_id: currentConv.id }
      });

      const response = await axios.post(`${API_BASE_URL}/api/v1/run`, {
        original_input: userMessage,
        graph_config_name: graphConfigToUse,
        initial_metadata: { conversation_id: currentConv.id }
      });

      const taskId: string = response.data.task_id;
      console.log('[sendMessage] HTTP POST success. Received taskId:', taskId);

      dispatch({ type: 'SET_TASK_ID', payload: { conversationId: currentConv.id, taskId } });

      // 이전 WebSocket 연결이 있다면 정리
      if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
        console.log('[sendMessage] Closing existing WebSocket before opening new one.');
        socketRef.current.close();
      }


      // const wsBaseUrl = API_BASE_URL.replace(/^http/, 'ws');
      // const wsUrl = `<span class="math-inline">\{wsBaseUrl\}</span>{settings.API_PREFIX}/ws/status/${taskId}`; /
      const wsUrl = `ws://localhost:8000/api/v1/ws/status/${taskId}`;
      console.log('[sendMessage] Attempting to connect to WebSocket URL:', wsUrl);

      const socket = new WebSocket(wsUrl);
      socketRef.current = socket; // ref에 현재 소켓 저장

      socket.onopen = () => {
        console.log('[WebSocket onopen] Connection successful for taskId:', taskId);
        setIsSending(false); // WebSocket 연결 성공 시 로딩 종료 (또는 첫 메시지 수신 시)
        dispatch({
            type: 'STATUS_UPDATE',
            payload: {
              conversationId: currentConv.id,
              status: 'connected_ws', // 웹소켓 연결 상태 명시
              detail: `WebSocket 연결됨 (Task: ${taskId})`
            }
        });
      };

      socket.onmessage = (event) => {
        console.log('[WebSocket onmessage] Raw data received:', event.data);
        try {
          const msg = JSON.parse(event.data as string);
          console.log('[WebSocket onmessage] Parsed message:', msg);

          if (!currentConv) {
            console.error('[WebSocket onmessage] CRITICAL: currentConv is undefined. Message:', msg);
            return;
          }
          // task_id가 메시지에 포함되어 있다면, 현재 대화의 task_id와 일치하는지 확인 (선택적)
          if (msg.task_id && msg.task_id !== currentConv.taskId) {
            console.warn(`[WebSocket onmessage] Received message for different taskId. Current: ${currentConv.taskId}, Received: ${msg.task_id}. Ignoring.`);
            return;
          }


          const eventType = msg.event_type;
          switch (eventType) {
            case 'status_update':
              console.log('[WebSocket onmessage] Dispatching STATUS_UPDATE:', msg);
              dispatch({
                type: 'STATUS_UPDATE',
                payload: {
                  conversationId: currentConv.id,
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
                  conversationId: currentConv.id,
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
                  payload: { conversationId: currentConv.id, finalAnswer: msg.final_answer }
                });
              }
              if (msg.error_message) { // final_result에 에러가 같이 올 수 있음
                dispatch({
                  type: 'FINAL_RESULT', // 또는 'ERROR' 액션 타입
                  payload: { conversationId: currentConv.id, errorMessage: msg.error_message }
                });
              }
              console.log('[WebSocket onmessage] Closing socket after final_result.');
              socket.close(1000, "Task completed"); // 정상 종료 코드 1000
              socketRef.current = null; // ref 초기화
              setIsSending(false); // 최종 결과 후 로딩 상태 해제
              break;
            case 'error': // WebSocket 프로토콜 상의 에러 메시지 (websocket_models.py에 정의됨)
              console.log('[WebSocket onmessage] Dispatching ERROR from server message:', msg);
              dispatch({
                type: 'ERROR',
                payload: { conversationId: currentConv.id, errorMessage: msg.message }
              });
              console.log('[WebSocket onmessage] Closing socket after server error message.');
              socket.close(1000, "Server error reported");
              socketRef.current = null;
              setIsSending(false);
              break;
            default:
              console.warn('[WebSocket onmessage] Unknown event_type:', eventType, 'Full message:', msg);
          }
        } catch (e) {
          console.error('[WebSocket onmessage] Error parsing JSON or dispatching:', e, 'Original data:', event.data);
          if (currentConv) {
            dispatch({
              type: 'ERROR',
              payload: { conversationId: currentConv.id, errorMessage: '수신 데이터 처리 중 오류 발생' }
            });
          }
          setIsSending(false);
        }
      };

      socket.onerror = (errorEvent) => {
        console.error('[WebSocket onerror] WebSocket error occurred:', errorEvent, 'URL attempted:', wsUrl);
        setIsSending(false); // 에러 발생 시 로딩 종료
        if (currentConv) {
          dispatch({
            type: 'ERROR',
            payload: { conversationId: currentConv.id, errorMessage: 'WebSocket 연결 중 심각한 오류 발생' }
          });
        }
        socketRef.current = null; // 에러 시 ref 초기화
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
        {currentConv ? (
          <>
            {currentConv.messages.map((msg, idx) => (
              <div key={idx} className={`message ${msg.sender}`}>
                <strong>{msg.sender === 'user' ? 'User' : 'Assistant'}:</strong> {msg.content}
              </div>
            ))}
            {/* 로딩 상태 또는 작업 상태 메시지 */}
            {(isSending || (currentConv.status && !['completed', 'failed', 'error', 'disconnected_ws'].includes(currentConv.status))) && (
              <div className="message assistant">
                <em>
                  {currentConv.status === 'connected_ws' || currentConv.status === 'pending_connection' ? '연결 중...' :
                   currentConv.status === 'running' || currentConv.status === 'status_update' ? `작업 진행 중 (${currentConv.statusDetail || currentConv.status})...` :
                   isSending ? '요청 전송 중...' :
                   `상태: ${currentConv.status}...`}
                </em>
              </div>
            )}
            {/* 오류 발생 시 오류 메시지 표시 */}
            {(currentConv.status === 'failed' || currentConv.status === 'error' || currentConv.status === 'disconnected_ws') && currentConv.errorMessage && (
              <div className="message error">
                <strong>Error:</strong> {currentConv.errorMessage}
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
          disabled={isSending || !currentConv} // 로딩 중이거나 대화 없을 시 비활성화
        />
        <button onClick={sendMessage} disabled={isSending || !currentConv || inputText.trim() === ''}>
          {isSending ? '전송 중...' : '전송'}
        </button>
      </div>
    </div>
  );
};

export default ChatWindow;