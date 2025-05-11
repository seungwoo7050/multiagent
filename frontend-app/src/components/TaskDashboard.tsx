import React, { useContext } from 'react';
import { AppContext } from '../context/ChatContext';

const TaskDashboard: React.FC = () => {
  const { state } = useContext(AppContext)!;
  const conv = state.conversations.find(c => c.id === state.currentConversationId);

  if (!conv) {
    return <div className="task-dashboard">진행 중인 Task 없음</div>;
  }

  return (
    <div className="task-dashboard">
      <h3>Task 진행 상태</h3>
      {/* 현재 상태 및 세부 설명 */}
      {conv.status && (
        <p>상태: <strong>{conv.status}</strong> {conv.statusDetail && `(${conv.statusDetail})`}</p>
      )}
      {conv.currentNode && <p>현재 노드: {conv.currentNode}</p>}
      {conv.nextNode && <p>다음 예정 노드: {conv.nextNode}</p>}

      {/* 중간 결과 목록 */}
      {conv.intermediateResults && conv.intermediateResults.length > 0 && (
        <>
          <h4>중간 결과</h4>
          <ul>
            {conv.intermediateResults.map((res, idx) => (
              <li key={idx}>
                <strong>{res.resultStepName}</strong>: {JSON.stringify(res.data)}
              </li>
            ))}
          </ul>
        </>
      )}

      {/* 최종 상태 표시 */}
      {conv.status === 'completed' && (
        <p><strong>✅ 워크플로우 완료!</strong></p>
      )}
      {conv.status === 'failed' && conv.errorMessage && (
        <p className="error"><strong>❌ 오류 발생:</strong> {conv.errorMessage}</p>
      )}
    </div>
  );
};

export default TaskDashboard;
