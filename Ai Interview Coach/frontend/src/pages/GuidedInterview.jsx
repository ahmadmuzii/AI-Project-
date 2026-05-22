import { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import { useParams, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import GuidedSetup from '../components/GuidedSetup';
import ChatSession from '../components/ChatSession';
import LiveInterviewSession from '../components/LiveInterviewSession';
import GuidedSummary from '../components/GuidedSummary';
import Navbar from '../components/Navbar';
import { ElevenLabsSpeaker } from '../utils/ElevenLabsSpeaker';
import { primeAudio } from '../utils/audioUnlock';
import {
  startGuidedInterview,
  answerClarification,
  answerGuidedQuestion,
  getGuidedInterview,
  endGuidedInterview,
} from '../api/client';

export default function GuidedInterviewPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { user, profile, refreshProfile } = useAuth();

  const [phase, setPhase] = useState('setup');
  const [interviewId, setInterviewId] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [totalEstimated, setTotalEstimated] = useState(0);
  const [answersSoFar, setAnswersSoFar] = useState(0);
  const [remainingSeconds, setRemainingSeconds] = useState(0);
  const [timeExpired, setTimeExpired] = useState(false);
  const [interviewResult, setInterviewResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sessionLoading, setSessionLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('text');
  const [avatarImage, setAvatarImage] = useState(null);
  const [useElevenLabs, setUseElevenLabs] = useState(false);
  const [elevenlabsVoiceId, setElevenlabsVoiceId] = useState('');
  const [greetingMessage, setGreetingMessage] = useState('');
  const [clarifyingQuestions, setClarifyingQuestions] = useState([]);
  const [clarifyingIndex, setClarifyingIndex] = useState(0);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [ttsError, setTtsError] = useState(null);
  const [clarifyingSubmitting, setClarifyingSubmitting] = useState(false);
  const elevenlabsRef = useRef(null);
  // Stable ref so speakText callback always sees latest ElevenLabsSpeaker instance
  const elevenlabsInstanceRef = useRef(null);
  const synthRef = useRef(null);
  // Track whether we've done a user-gesture audio unlock for this session
  const audioUnlockedRef = useRef(false);

  useEffect(() => {
    if (id) {
      setLoading(true);
      getGuidedInterview(parseInt(id))
        .then((result) => {
          if (result.interview.status === 'completed') {
            setInterviewResult(result);
            setPhase('summary');
          } else if (result.interview.phase === 'greeting') {
            setMode(result.interview.mode || 'text');
            setInterviewId(result.interview.id);
            setGreetingMessage(result.interview.greeting_message || '');
            setClarifyingQuestions(result.interview.clarifying_questions || []);
            const answers = result.interview.clarification_answers || [];
            setClarifyingIndex(answers.length);
            setPhase('greeting');
          } else if (result.interview.status === 'in_progress') {
            setMode(result.interview.mode || 'text');
            setInterviewId(result.interview.id);
            setTotalEstimated(999);
            setRemainingSeconds(result.remaining_seconds);
            setTimeExpired(result.time_expired);
            if (result.pending_question) {
              setCurrentQuestion(result.pending_question.question);
              setAnswersSoFar(result.qa_pairs.filter((q) => q.transcript).length);
            }
            setPhase('session');
          }
        })
        .catch(() => navigate('/interview/new'))
        .finally(() => setLoading(false));
    }
  }, [id]);

  const handleStart = async (details) => {
    setSessionLoading(true);
    setMode(details.mode || 'text');
    if (details.avatar_image) setAvatarImage(details.avatar_image);
    setUseElevenLabs(!!details.use_elevenlabs);
    if (details.elevenlabs_voice_id) setElevenlabsVoiceId(details.elevenlabs_voice_id);

    // Init ElevenLabs immediately for live mode so child can use it on mount
    if (details.mode === 'live' && details.use_elevenlabs && details.elevenlabs_voice_id) {
      if (!elevenlabsRef.current) {
        const speaker = new ElevenLabsSpeaker({
          voiceId: details.elevenlabs_voice_id,
          playbackRate: 1.15,
          onStart: () => setIsSpeaking(true),
          onEnd: () => setIsSpeaking(false),
          onError: (err) => {
            console.warn('TTS error:', err);
            setTtsError('Voice playback failed — check console for details');
            setTimeout(() => setTtsError(null), 5000);
          },
        });
        elevenlabsRef.current = speaker;
        elevenlabsInstanceRef.current = speaker;
      }
    }
    // Unlock BOTH AudioContext AND HTML5 Audio on the Begin Interview click (user gesture)
    // primeAudio() handles both - critical for ElevenLabs which uses new Audio()
    primeAudio();
    audioUnlockedRef.current = true;

    try {
      const result = await startGuidedInterview({
        aim: details.aim,
        target_company: details.target_company,
        duration_minutes: details.duration_minutes,
        difficulty: details.difficulty,
        focus_areas: details.focus_areas,
        mode: details.mode || 'text',
      });
      setInterviewId(result.interview_id);
      setGreetingMessage(result.greeting_message);
      setClarifyingQuestions(result.clarifying_questions);
      setClarifyingIndex(0);

      if (details.mode === 'text' || details.mode === 'live') {
        // Both modes: auto-answer clarifying questions — user already gave all context in setup form
        let lastResult = null;
        for (let i = 0; i < (result.clarifying_questions || []).length; i++) {
          lastResult = await answerClarification(result.interview_id, i, null, '');
        }
        if (lastResult && lastResult.first_question) {
          setCurrentQuestion(lastResult.first_question);
          setTotalEstimated(999);
          if (details.mode === 'live' && result.greeting_message) {
            setPhase('session');
            navigate(`/interview/${result.interview_id}`, { replace: true });
            setTimeout(() => {
              speakTextForStart(`${result.greeting_message} Here is your first question: ${lastResult.first_question}`);
            }, 800);
          } else {
            setPhase('session');
            navigate(`/interview/${result.interview_id}`, { replace: true });
          }
        } else {
          setPhase('greeting');
        }
      }
    } catch (err) {
      console.error('Failed to start interview:', err);
      setError(err.message || 'Failed to start interview. Check server is running.');
      setTimeout(() => setError(null), 5000);
    } finally {
      setSessionLoading(false);
    }
  };

  // ElevenLabs / TTS setup — only for live mode
  useEffect(() => {
    if (mode !== 'live') return;
    if (useElevenLabs && elevenlabsVoiceId) {
      if (!elevenlabsRef.current) {
        const speaker = new ElevenLabsSpeaker({
          voiceId: elevenlabsVoiceId,
          playbackRate: 1.15,
          onStart: () => setIsSpeaking(true),
          onEnd: () => setIsSpeaking(false),
          onError: (err) => {
            console.warn('TTS error:', err);
            setTtsError('Voice playback failed — check console for details');
            setTimeout(() => setTtsError(null), 5000);
          },
        });
        elevenlabsRef.current = speaker;
        elevenlabsInstanceRef.current = speaker;
      } else {
        elevenlabsRef.current.setVoice(elevenlabsVoiceId);
        elevenlabsInstanceRef.current = elevenlabsRef.current;
      }
    }
    return () => {
      if (elevenlabsRef.current) elevenlabsRef.current.stop();
    };
  }, [mode, useElevenLabs, elevenlabsVoiceId]);

  // speakText: always read the instance from the ref to avoid stale closure
  const speakText = useCallback((text) => {
    if (!text) return;
    // Ensure audio is unlocked
    if (!audioUnlockedRef.current) {
      primeAudio();
      audioUnlockedRef.current = true;
    }
    const instance = elevenlabsInstanceRef.current;
    if (useElevenLabs && instance) {
      instance.speak(text);
      return;
    }
    if (!window.speechSynthesis) {
      setTtsError('Browser speech synthesis not available');
      setTimeout(() => setTtsError(null), 4000);
      return;
    }
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.0;
    u.volume = 1;
    u.onstart = () => setIsSpeaking(true);
    u.onend = () => setIsSpeaking(false);
    u.onerror = (e) => {
      console.warn('SpeechSynthesis error:', e);
      setTtsError('Browser voice failed to play');
      setTimeout(() => setTtsError(null), 4000);
      setIsSpeaking(false);
    };
    window.speechSynthesis.speak(u);
  }, [useElevenLabs]); // elevenlabsInstanceRef is a ref — stable, no need in deps

  // speakTextForStart: called immediately after interview starts — reads directly from refs
  // to avoid any stale closure issue with useCallback during the very first speak
  const speakTextForStart = (text) => {
    if (!text) return;
    const instance = elevenlabsInstanceRef.current;
    if (instance) {
      instance.speak(text);
      return;
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 1.0;
      u.onstart = () => setIsSpeaking(true);
      u.onend = () => setIsSpeaking(false);
      window.speechSynthesis.speak(u);
    }
  };

  const handleClarificationAnswer = async (blob) => {
    try {
      const result = await answerClarification(interviewId, clarifyingIndex, blob, '');
      if (result.done) {
        setCurrentQuestion(result.first_question);
        setTotalEstimated(result.total_estimated);
        setPhase('session');
        setClarifyingIndex(0);
      } else {
        setClarifyingIndex((prev) => prev + 1);
      }
      return result;
    } catch (err) {
      console.error('Clarification answer failed:', err);
      setError(err.message || 'Failed to process your answer.');
      setTimeout(() => setError(null), 5000);
    }
  };

  const handleAnswer = async (input) => {
    setSessionLoading(true);
    try {
      const isText = typeof input === 'string';
      const result = await answerGuidedQuestion(interviewId, isText ? null : input, isText ? input : '');
      setAnswersSoFar((prev) => prev + 1);
      if (result.is_complete) {
        const interviewData = await getGuidedInterview(interviewId);
        setInterviewResult(interviewData);
        setPhase('summary');
        navigate(`/interview/${interviewId}`, { replace: true });
        refreshProfile();
        return result;
      }
      if (result.next_question) setCurrentQuestion(result.next_question);
      return result;
    } catch (err) {
      console.error('Failed to submit answer:', err);
      throw err;
    } finally {
      setSessionLoading(false);
    }
  };

  const handleEnd = async () => {
    setSessionLoading(true);
    try {
      await endGuidedInterview(interviewId);
      const interviewData = await getGuidedInterview(interviewId);
      setInterviewResult(interviewData);
      setPhase('summary');
      navigate(`/interview/${interviewId}`, { replace: true });
      refreshProfile();
    } catch (err) {
      console.error('Failed to end interview:', err);
    } finally {
      setSessionLoading(false);
    }
  };

  const handleRetry = () => {
    setPhase('setup');
    setInterviewId(null);
    setCurrentQuestion('');
    setInterviewResult(null);
    setAnswersSoFar(0);
    setAvatarImage(null);
    setGreetingMessage('');
    setClarifyingQuestions([]);
    setClarifyingIndex(0);
    navigate('/interview/new', { replace: true });
  };

  if (loading) {
    return (
      <div>
        <Navbar />
        <div className="flex center" style={{ padding: 120 }}>
          <div className="spinner" />
        </div>
      </div>
    );
  }

  return (
    <div>
      <Navbar />

      {ttsError && (
        <div style={{
          position: 'fixed', top: 72, left: '50%', transform: 'translateX(-50%)',
          zIndex: 9999, padding: '10px 20px', borderRadius: 12,
          background: 'var(--red-bg, #fce4ec)', color: 'var(--red, #c62828)',
          border: '1px solid var(--red, #ef5350)',
          fontSize: '0.85rem', fontWeight: 500,
          boxShadow: '0 4px 16px rgba(0,0,0,0.15)',
          pointerEvents: 'none',
        }}>
          {ttsError}
        </div>
      )}

      {phase === 'setup' && (
        <>
          {error && (
            <div style={{
              maxWidth: 640, margin: '80px auto 0', padding: '0 24px',
            }}>
              <div className="card" style={{
                borderLeft: '3px solid var(--red)', color: 'var(--red)',
                fontSize: '0.9rem', padding: '12px 16px',
              }}>
                {error}
              </div>
            </div>
          )}
          <GuidedSetup profile={profile} onStart={handleStart} loading={sessionLoading} />
        </>
      )}

      {(phase === 'greeting' || phase === 'session') && mode === 'live' && (
        <LiveInterviewSession
          interviewId={interviewId}
          currentQuestion={currentQuestion}
          phase={phase}
          onAnswer={handleAnswer}
          onEnd={handleEnd}
          loading={sessionLoading}
          totalEstimated={totalEstimated}
          answersSoFar={answersSoFar}
          remainingSeconds={remainingSeconds}
          timeExpired={timeExpired}
          avatarImage={avatarImage}
          useElevenLabs={useElevenLabs}
          elevenlabsVoiceId={elevenlabsVoiceId}
          elevenlabsRef={elevenlabsRef}
          speakText={speakText}
          isSpeaking={isSpeaking}
          greetingMessage={greetingMessage}
          clarifyingQuestions={clarifyingQuestions}
          clarifyingIndex={clarifyingIndex}
          onClarificationAnswer={handleClarificationAnswer}
        />
      )}
      {phase === 'session' && mode === 'text' && (
        <ChatSession
          interviewId={interviewId}
          currentQuestion={currentQuestion}
          onAnswer={handleAnswer}
          onEnd={handleEnd}
          loading={sessionLoading}
          totalEstimated={totalEstimated}
          answersSoFar={answersSoFar}
          remainingSeconds={remainingSeconds}
          timeExpired={timeExpired}
        />
      )}
      {phase === 'summary' && interviewResult && (
        <GuidedSummary interview={interviewResult} onRetry={handleRetry} />
      )}
    </div>
  );
}
