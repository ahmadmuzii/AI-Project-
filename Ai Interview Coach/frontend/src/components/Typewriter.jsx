import useTypewriter from '../hooks/useTypewriter';

export default function Typewriter({ text, speed = 50, delay = 0, startWhen = true, className = '' }) {
  const { displayText, isTyping, isDone } = useTypewriter(text, { speed, delay, startTyping: startWhen });

  return (
    <span className={className}>
      {displayText}
      <span className={`typewriter-cursor ${isDone ? 'done' : ''} ${isTyping ? 'typing' : ''}`}>|</span>
    </span>
  );
}
