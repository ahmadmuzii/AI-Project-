import { useRef } from 'react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform, useInView } from 'framer-motion';
import Typewriter from '../components/Typewriter';
import useMagnetic from '../hooks/useMagnetic';
import './Landing.css';

const featureData = [
  { icon: '🎙️', title: 'Speech Analysis', desc: 'Measures pace, clarity, filler words, pitch variation, and vocal energy with millisecond precision.' },
  { icon: '🧠', title: 'AI Coaching Feedback', desc: 'Get personalized, actionable feedback powered by Groq Llama 3.3 with rule-based fallback analysis.' },
  { icon: '🎯', title: 'Live Practice Mode', desc: 'Simulate real interviews with spoken questions, adaptive difficulty, and eye tracking via WebRTC.' },
  { icon: '📄', title: 'Resume Analyzer', desc: 'Upload your PDF resume and get a skill gap analysis matched against your target role.' },
  { icon: '📈', title: 'Progress Dashboard', desc: 'Track your improvement over time with charts, streaks, leaderboards, and weak-topic heatmaps.' },
  { icon: '🏢', title: 'Company Mode', desc: 'Practice with questions tailored to top companies — Google, Meta, McKinsey, and more.' },
];

const techData = [
  { icon: '🎤', title: 'Speech Recognition', desc: 'High-accuracy transcription powered by Whisper, handling diverse accents and audio conditions in real time.' },
  { icon: '🧠', title: 'AI-Powered Coaching', desc: 'Intelligent feedback engine analyzing content structure, relevance, and delivery using advanced LLMs.' },
  { icon: '📊', title: 'Real-Time Analytics', desc: 'Extracts 40+ acoustic and linguistic features — pace, filler words, pitch variation, and vocal energy.' },
  { icon: '📄', title: 'Resume Intelligence', desc: 'Parses PDF resumes and performs skill gap analysis against target roles and industry benchmarks.' },
  { icon: '📈', title: 'Progress Tracking', desc: 'Tracks improvement over time with detailed charts, streaks, leaderboard rankings, and weak-topic heatmaps.' },
  { icon: '🏢', title: 'Company-Specific Prep', desc: 'Tailored question banks for top companies — Google, Meta, McKinsey, and more — with adaptive difficulty.' },
];

const stepsData = [
  { num: '01', icon: '🎙️', title: 'Record Your Answer', desc: 'Use your microphone or upload an audio file. Speak naturally as you would in a real interview.' },
  { num: '02', icon: '🤖', title: 'AI Analyzes Everything', desc: 'Your speech is transcribed and analyzed for pace, clarity, and vocal variety. NLP evaluates your content structure and relevance.' },
  { num: '03', icon: '📈', title: 'Get Actionable Feedback', desc: 'Review your transcript, scores, word-level analysis, and AI coaching tips. Track your growth over time.' },
];

function SectionHeader({ title, desc, light, typewriterText }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });
  return (
    <motion.div
      ref={ref}
      className="section-header"
      initial={{ opacity: 0, y: 48 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
    >
      <h2 className="section-title-text">
        {typewriterText ? (
          <Typewriter text={typewriterText} speed={40} startWhen={inView} />
        ) : title}
      </h2>
      {desc && <p className="section-desc-text">{desc}</p>}
    </motion.div>
  );
}

function FeatureCard({ icon, title, desc, index }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });
  return (
    <motion.div
      ref={ref}
      className="glass-card feature-card"
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7, delay: index * 0.08, ease: [0.16, 1, 0.3, 1] }}
    >
      <span className="feature-icon">{icon}</span>
      <h3 className="feature-title">{title}</h3>
      <p className="feature-desc">{desc}</p>
    </motion.div>
  );
}

function TechCard({ icon, title, desc, index }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });
  return (
    <motion.div
      ref={ref}
      className="glass-card tech-card"
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay: index * 0.06, ease: [0.16, 1, 0.3, 1] }}
    >
      <span className="tech-icon">{icon}</span>
      <h3 className="tech-title">{title}</h3>
      <p className="tech-desc">{desc}</p>
    </motion.div>
  );
}

function StepCard({ num, icon, title, desc, index }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });
  return (
    <motion.div
      ref={ref}
      className="step-card"
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.8, delay: index * 0.2, ease: [0.16, 1, 0.3, 1] }}
    >
      <span className="step-num">{num}</span>
      <span className="step-icon">{icon}</span>
      <h3 className="step-title">{title}</h3>
      <p className="step-desc">{desc}</p>
    </motion.div>
  );
}

function Magnetic({ children, className = '' }) {
  const { ref, pos, handleMouseMove, handleMouseLeave } = useMagnetic();
  return (
    <motion.div
      ref={ref}
      className={className}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      animate={{ x: pos.x, y: pos.y }}
      transition={{ type: 'spring', stiffness: 150, damping: 15, mass: 0.5 }}
    >
      {children}
    </motion.div>
  );
}

export default function Landing() {
  const heroRef = useRef(null);
  const footerRef = useRef(null);
  const footerInView = useInView(footerRef, { once: true, margin: '-80px' });
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ['start start', 'end start'] });
  const heroScale = useTransform(scrollYProgress, [0, 1], [1, 0.75]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);

  return (
    <div className="landing">
      {/* Nav */}
      <nav className="landing-nav">
        <div className="landing-nav-inner">
          <span className="landing-nav-logo">AI Interview Coach</span>
          <div className="landing-nav-links">
            <a href="#features" className="landing-nav-link">Features</a>
            <a href="#how" className="landing-nav-link">How it Works</a>
          </div>
          <Magnetic><Link to="/login" className="landing-nav-cta">Get Started</Link></Magnetic>
        </div>
      </nav>

      {/* Hero */}
      <section ref={heroRef} className="landing-hero">
        <motion.div className="landing-hero-inner" style={{ scale: heroScale, opacity: heroOpacity }}>
          <div className="hero-left">
            <p className="hero-label">AI Interview Coach</p>
            <h1 className="hero-title">
              <Typewriter text="Practice smarter." speed={50} startWhen={true} />
              <br />
              <Typewriter text="Get hired faster." speed={50} delay={1400} startWhen={true} />
            </h1>
            <p className="hero-sub">
              Personalized AI interviews with real-time feedback, confidence scoring, and behavioral analysis.
            </p>
            <div className="hero-ctas">
              <Magnetic><Link to="/login" className="hero-btn-primary">Get Started</Link></Magnetic>
              <Magnetic><a href="#features" className="hero-btn-secondary">Learn More</a></Magnetic>
            </div>
          </div>
          <div className="hero-right">
            <div className="hero-card">
              <div className="hero-card-dots">
                <span className="dot-red" />
                <span className="dot-yellow" />
                <span className="dot-green" />
              </div>
              <div className="hero-card-body">
                <div className="hc-row hc-question">
                  <p className="hc-label">Current Question</p>
                  <p className="hc-value">Tell me about yourself.</p>
                </div>
                <div className="hc-row hc-ai">
                  <p className="hc-label-light">AI Feedback</p>
                  <p className="hc-value-light">Great confidence level. Try shortening your introduction.</p>
                </div>
                <div className="hc-row hc-confidence">
                  <p className="hc-label">Confidence Score</p>
                  <div className="hc-bar">
                    <div className="hc-bar-fill" style={{ width: '82%' }} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Features */}
      <section id="features" className="section-block section-features">
        <SectionHeader title="Everything you need to ace the interview." typewriterText="Everything you need to ace the interview." desc="AI-powered tools to analyze, improve, and track your performance." />
        <div className="grid-3">
          {featureData.map((f, i) => (
            <FeatureCard key={f.title} {...f} index={i} />
          ))}
        </div>
      </section>

      {/* Tech */}
      <section className="section-block section-tech">
        <SectionHeader title="Built with cutting-edge AI." typewriterText="Built with cutting-edge AI." desc="Powered by industry-leading machine learning and web frameworks." light />
        <div className="grid-3">
          {techData.map((t, i) => (
            <TechCard key={t.title} {...t} index={i} />
          ))}
        </div>
      </section>

      {/* How it Works */}
      <section id="how" className="section-block section-how">
        <SectionHeader title="How it works." typewriterText="How it works." desc="Three simple steps to interview mastery." />
        <div className="steps-row">
          {stepsData.map((s, i) => (
            <StepCard key={s.num} {...s} index={i} />
          ))}
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="landing-footer-content">
          <motion.h2
            ref={footerRef}
            className="footer-title"
            initial={{ opacity: 0, y: 60 }}
            animate={footerInView ? { opacity: 1, y: 0 } : {}}
            transition={{ duration: 0.8, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          >
            <Typewriter text="Start your practice today." speed={45} startWhen={footerInView} />
          </motion.h2>

          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            <Magnetic><Link to="/login" className="footer-cta">Get Started</Link></Magnetic>
          </motion.div>
          <motion.div
            className="footer-divider"
            initial={{ scaleX: 0 }}
            whileInView={{ scaleX: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
          />
          <motion.div
            className="footer-bottom"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.6, ease: [0.16, 1, 0.3, 1] }}
          >
            <p className="footer-copy">&copy; 2026 AI Interview Coach. All rights reserved.</p>
            <div className="footer-links">
              <a href="https://github.com/ahmadmuzii" target="_blank" rel="noopener noreferrer">GitHub</a>
              <a href="mailto:ahmadmuzi@gmail.com">Contact</a>
              <a href="#">Project Docs</a>
            </div>
          </motion.div>
        </div>
      </footer>
    </div>
  );
}
