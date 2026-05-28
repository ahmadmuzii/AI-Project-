import { useState, useEffect, useRef, useCallback } from 'react';

export default function useTypewriter(text, { speed = 50, delay = 0, startTyping = true } = {}) {
  const [displayText, setDisplayText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isDone, setIsDone] = useState(false);
  const indexRef = useRef(0);
  const timerRef = useRef(null);
  const delayTimerRef = useRef(null);

  const reset = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (delayTimerRef.current) clearTimeout(delayTimerRef.current);
    indexRef.current = 0;
    setDisplayText('');
    setIsTyping(false);
    setIsDone(false);
  }, []);

  useEffect(() => {
    reset();
    if (!text || !startTyping) return;

    delayTimerRef.current = setTimeout(() => {
      setIsTyping(true);
      timerRef.current = setInterval(() => {
        indexRef.current += 1;
        if (indexRef.current >= text.length) {
          setDisplayText(text);
          setIsTyping(false);
          setIsDone(true);
          if (timerRef.current) clearInterval(timerRef.current);
        } else {
          setDisplayText(text.slice(0, indexRef.current));
        }
      }, speed);
    }, delay);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (delayTimerRef.current) clearTimeout(delayTimerRef.current);
    };
  }, [text, speed, delay, startTyping, reset]);

  return { displayText, isTyping, isDone };
}
