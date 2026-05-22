const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function getToken() {
  try {
    const stored = localStorage.getItem('aic_auth');
    if (stored) return JSON.parse(stored).access_token;
  } catch {}
  return null;
}

function authHeaders() {
  const token = getToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function request(method, path, { json, form, auth } = {}) {
  const opts = { method, headers: { ...(auth ? authHeaders() : {}) } };
  if (json) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(json);
  } else if (form) {
    opts.body = form;
  }
  const res = await fetch(`${API}${path}`, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Server error (${res.status})`);
  }
  return res.json();
}

export const register = (name, email, password) =>
  request('POST', '/auth/register', { json: { name, email, password } });

export const login = (email, password) =>
  request('POST', '/auth/login', { json: { email, password } });

export const getMe = () =>
  request('GET', '/auth/me', { auth: true });

export const createUser = (name, email) =>
  request('POST', '/create-user', { json: { name, email } });

export const startSession = () =>
  request('POST', '/start-session', { auth: true });

export const listSessions = () =>
  request('GET', '/sessions', { auth: true });

export const clearSessions = () =>
  request('DELETE', '/sessions', { auth: true });

export const listRecordings = (session_id) =>
  request('GET', `/recordings/${session_id}`, { auth: true });

export const recordingAudioUrl = (recording_id) =>
  `${API}/recording/${recording_id}/audio`;

export const deleteRecording = (recording_id) =>
  request('DELETE', `/recording/${recording_id}`, { auth: true });

export const uploadAudio = (session_id, file, role = 'general', topic = 'general', question = '', resume_text = '') => {
  const form = new FormData();
  form.append('session_id', String(session_id));
  form.append('role', role);
  form.append('topic', topic);
  form.append('question', question);
  form.append('resume_text', resume_text);
  form.append('file', file, file.name || 'audio.webm');  // was missing — caused 422 errors
  return request('POST', '/upload-audio', { form, auth: true });
};

export const getDashboard = () =>
  request('GET', '/dashboard', { auth: true });

export const getLeaderboard = (role) =>
  request('GET', `/leaderboard?role=${role}`, { auth: true });

export const nlpAnalyzeAnswer = (answer, role) => {
  const form = new FormData();
  form.append('answer', answer);
  form.append('role', role);
  return request('POST', '/nlp/analyze-answer', { form, auth: true });
};

export const adaptiveNextQuestions = (role, weak_topics, previous_questions) => {
  const form = new FormData();
  form.append('role', role);
  form.append('weak_topics', weak_topics.join(','));
  form.append('previous_questions', previous_questions.join('||'));
  return request('POST', '/adaptive/next-questions', { form, auth: true });
};

export const resumeAnalyze = (resume_text, role) => {
  const form = new FormData();
  form.append('resume_text', resume_text);
  form.append('role', role);
  return request('POST', '/resume/analyze', { form, auth: true });
};

export const companyMode = (company, role) =>
  request('GET', `/company-mode?company=${company}&role=${role}`, { auth: true });

export const studyPlan = () =>
  request('GET', '/study-plan', { auth: true });

export const stressEvaluate = (eye, move, voice) => {
  const form = new FormData();
  form.append('eye_contact_score', String(eye));
  form.append('movement_score', String(move));
  form.append('voice_energy', String(voice));
  return request('POST', '/stress/evaluate', { form, auth: true });
};

export const analyzeWebcamFrame = (frameBlob, streamId, voiceEnergy) => {
  const form = new FormData();
  form.append('frame', frameBlob, 'webcam.jpg');
  form.append('stream_id', streamId);
  form.append('voice_energy', String(voiceEnergy));
  return request('POST', '/stress/analyze-webcam', { form, auth: true });
};

// ── Profile ──

export const getProfile = () =>
  request('GET', '/auth/profile', { auth: true });

export const updateProfile = (data) => {
  const form = new FormData();
  for (const [key, val] of Object.entries(data)) {
    if (val !== undefined && val !== null) {
      if (key === 'focus_areas' && Array.isArray(val)) {
        form.append(key, JSON.stringify(val));
      } else if (key === 'upcoming_interview_date' && val) {
        form.append(key, val instanceof Date ? val.toISOString().split('T')[0] : val);
      } else {
        form.append(key, String(val));
      }
    }
  }
  return request('PUT', '/auth/profile', { form, auth: true });
};

export const uploadAvatar = (file) => {
  const form = new FormData();
  form.append('file', file, file.name);
  return request('POST', '/auth/upload-avatar', { form, auth: true });
};

export const uploadResume = (file) => {
  const form = new FormData();
  form.append('file', file, file.name);
  return request('POST', '/auth/upload-resume', { form, auth: true });
};

export const changePassword = (old_password, new_password) =>
  request('POST', '/auth/change-password', { json: { old_password, new_password }, auth: true });

export const deleteAccount = (password) =>
  request('POST', '/auth/delete-account', { json: { password }, auth: true });

// ── Resume ──

export const listResumes = () =>
  request('GET', '/resume/list', { auth: true });

export const uploadResumeFile = (file, name = 'Untitled Resume') => {
  const form = new FormData();
  form.append('file', file, file.name);
  form.append('name', name);
  return request('POST', '/resume/upload', { form, auth: true });
};

export const createManualResume = (data) => {
  const form = new FormData();
  form.append('name', data.name || 'Manual Entry');
  form.append('skills', JSON.stringify(data.skills || []));
  form.append('experience_years', String(data.experience_years || 0));
  form.append('education', JSON.stringify(data.education || []));
  form.append('summary', data.summary || '');
  return request('POST', '/resume/manual', { form, auth: true });
};

export const setPrimaryResume = (resume_id) =>
  request('PUT', `/resume/${resume_id}/primary`, { auth: true });

export const deleteResume = (resume_id) =>
  request('DELETE', `/resume/${resume_id}`, { auth: true });

export const analyzeResumeProfile = (resume_id) =>
  request('GET', `/resume/${resume_id}/profile`, { auth: true });

export const analyzeResume = (resume_id, role) =>
  request('GET', `/resume/${resume_id}/analysis${role ? `?role=${role}` : ''}`, { auth: true });

// ── Guided Interview ──

export const startGuidedInterview = ({ aim, target_company, duration_minutes, difficulty, focus_areas, mode }) => {
  const form = new FormData();
  form.append('aim', aim || '');
  form.append('target_company', target_company || '');
  form.append('duration_minutes', String(duration_minutes || 30));
  form.append('difficulty', difficulty || 'intermediate');
  form.append('focus_areas', JSON.stringify(focus_areas || []));
  form.append('mode', mode || 'text');
  return request('POST', '/guided/start', { form, auth: true });
};

export const answerClarification = (interview_id, index, file, text) => {
  const form = new FormData();
  if (file) form.append('file', file, file.name || 'answer.wav');
  if (text) form.append('text', text);
  form.append('index', String(index));
  return request('POST', `/guided/${interview_id}/answer-clarification`, { form, auth: true });
};

export const answerGuidedQuestion = (interview_id, file, text) => {
  const form = new FormData();
  if (file) form.append('file', file, file.name || 'answer.wav');
  if (text) form.append('text', text);
  return request('POST', `/guided/${interview_id}/answer`, { form, auth: true });
};

export const listGuidedInterviews = () =>
  request('GET', '/guided/list', { auth: true });

export const getGuidedInterview = (interview_id) =>
  request('GET', `/guided/${interview_id}`, { auth: true });

export const endGuidedInterview = (interview_id) =>
  request('POST', `/guided/${interview_id}/end`, { auth: true });

export const uploadResumeGuided = (interview_id, file) => {
  const form = new FormData();
  form.append('file', file, file.name);
  return request('POST', `/guided/${interview_id}/upload-resume`, { form, auth: true });
};

// ── ElevenLabs ──

export const listElevenLabsVoices = () =>
  request('GET', '/api/elevenlabs/voices', { auth: true });

export const getElevenLabsTtsUrl = (text, voiceId) =>
  `${API}/api/elevenlabs/tts?text=${encodeURIComponent(text)}&voice_id=${voiceId}`;
