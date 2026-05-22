import { useRef } from 'react';
import { Link } from 'react-router-dom';
import { motion, useScroll, useTransform, useInView } from 'framer-motion';
import { useAuth } from '../context/AuthContext';
import './Landing.css';

function SectionHeader({ title, desc, light }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });
  return (
    <motion.div
      ref={ref}
      className="landing-section-header"
      initial={{ opacity: 0, y: 48 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      style={light ? { color: '#f5f5f7' } : {}}
    >
      <h2 className="landing-section-title">{title}</h2>
      {desc && <p className="landing-section-desc">{desc}</p>}
    </motion.div>
  );
}

const featureData = [
  { icon: 'icon-wave', title: 'Speech Analysis', desc: 'Measures pace, clarity, filler words, pitch variation, and vocal energy with millisecond precision.' },
  { icon: 'icon-brain', title: 'AI Coaching Feedback', desc: 'Get personalized, actionable feedback powered by Groq Llama 3.3 with rule-based fallback analysis.' },
  { icon: 'icon-live', title: 'Live Practice Mode', desc: 'Simulate real interviews with spoken questions, adaptive difficulty, and eye tracking via WebRTC.' },
  { icon: 'icon-resume', title: 'Resume Analyzer', desc: 'Upload your PDF resume and get a skill gap analysis matched against your target role.' },
  { icon: 'icon-chart', title: 'Progress Dashboard', desc: 'Track your improvement over time with charts, streaks, leaderboards, and weak-topic heatmaps.' },
  { icon: 'icon-company', title: 'Company Mode', desc: 'Practice with questions tailored to top companies — Google, Meta, McKinsey, and more.' },
];

const techData = [
  { name: 'Whisper', color: '#1A73E8', desc: 'OpenAI speech-to-text' },
  { name: 'PyTorch', color: '#EE4C2C', desc: 'Deep learning framework' },
  { name: 'FastAPI', color: '#009688', desc: 'High-performance Python API' },
  { name: 'librosa', color: '#8E24AA', desc: 'Audio feature extraction' },
  { name: 'SQLAlchemy', color: '#E91E63', desc: 'SQL ORM & database' },
  { name: 'React', color: '#61DAFB', desc: 'Interactive UI framework' },
  { name: 'Groq', color: '#F97316', desc: 'LLM inference (Llama 3.3)' },
  { name: 'NumPy', color: '#00BCD4', desc: 'Numerical computing' },
];

const stepsData = [
  { num: '01', icon: 'icon-mic-step', title: 'Record Your Answer', desc: 'Use your microphone or upload an audio file. Speak naturally as you would in a real interview.' },
  { num: '02', icon: 'icon-ai-step', title: 'AI Analyzes Everything', desc: 'Whisper transcribes your speech. librosa extracts 40+ acoustic features. NLP scores your content structure.' },
  { num: '03', icon: 'icon-growth-step', title: 'Get Actionable Feedback', desc: 'Review your transcript, scores, word-level analysis, and AI coaching tips. Track your growth over time.' },
];

function FeatureCard({ icon, title, desc, index }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });
  return (
    <motion.div
      ref={ref}
      className="landing-feature-card"
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7, delay: index * 0.08, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className={`landing-f-icon ${icon}`} />
      <h3>{title}</h3>
      <p>{desc}</p>
    </motion.div>
  );
}

function TechCard({ name, color, desc, index }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });
  return (
    <motion.div
      ref={ref}
      className="landing-tech-card"
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.5, delay: index * 0.06, ease: [0.16, 1, 0.3, 1] }}
    >
      <span className="landing-tech-badge" style={{ '--badge': color }}>{name}</span>
      <span className="landing-tech-desc">{desc}</span>
    </motion.div>
  );
}

function StepCard({ num, icon, title, desc, index }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });
  return (
    <motion.div
      ref={ref}
      className="landing-step"
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.8, delay: index * 0.2, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="landing-step-number">{num}</div>
      <div className={`landing-step-icon ${icon}`} />
      <h3>{title}</h3>
      <p>{desc}</p>
    </motion.div>
  );
}

export default function Landing() {
  const { user } = useAuth();
  const heroRef = useRef(null);
  const { scrollYProgress } = useScroll({ target: heroRef, offset: ['start start', 'end start'] });
  const heroScale = useTransform(scrollYProgress, [0, 1], [1, 0.75]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);

  return (
    <div className="landing">
      <nav className="landing-nav" id="landingNav">
        <div className="landing-nav-inner">
          <span className="landing-nav-logo">AI Interview Coach</span>
          {user ? (
            <Link to="/dashboard" className="landing-nav-btn">Dashboard</Link>
          ) : (
            <Link to="/login" className="landing-nav-btn">Get Started</Link>
          )}
        </div>
      </nav>

      <section ref={heroRef} className="landing-hero">
        <motion.div className="landing-hero-content" style={{ scale: heroScale, opacity: heroOpacity }}>
          <p className="landing-hero-label">AI-Powered Interview Preparation</p>
          <h1 className="landing-hero-title">Master Your<br />Next Interview</h1>
          <p className="landing-hero-subtitle">Record your answers. Get instant AI feedback. Improve faster.</p>
          <div className="landing-hero-device">
            <div className="landing-device-ring">
              <div className="landing-device-ring-inner">
                <div className="landing-device-mic" />
              </div>
            </div>
          </div>
          <Link to="/login" className="landing-hero-cta">Get Started — it's free</Link>
        </motion.div>
      </section>

      <section className="landing-section landing-features">
        <SectionHeader title="Everything you need to ace the interview." desc="AI-powered tools to analyze, improve, and track your performance." />
        <div className="landing-features-grid">
          {featureData.map((f, i) => (
            <FeatureCard key={f.title} {...f} index={i} />
          ))}
        </div>
      </section>

      <section className="landing-section landing-tech">
        <SectionHeader title="Built with cutting-edge AI." desc="Powered by industry-leading machine learning and web frameworks." light />
        <div className="landing-tech-grid">
          {techData.map((t, i) => (
            <TechCard key={t.name} {...t} index={i} />
          ))}
        </div>
      </section>

      <section className="landing-section landing-how">
        <SectionHeader title="How it works." desc="Three simple steps to interview mastery." />
        <div className="landing-steps">
          {stepsData.map((s, i) => (
            <StepCard key={s.num} {...s} index={i} />
          ))}
        </div>
      </section>

      <footer className="landing-section landing-footer">
        <div className="landing-footer-content">
          <motion.p
            className="landing-footer-label"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          >
            AI Interview Coach
          </motion.p>
          <motion.h2
            className="landing-footer-title"
            initial={{ opacity: 0, y: 60 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 1, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          >
            Start your<br />practice today.
          </motion.h2>
          <motion.p
            className="landing-footer-subtitle"
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
          >
            Semester project — AI, Spring 2026
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            <Link to="/login" className="landing-cta-button">Get Started</Link>
          </motion.div>
          <motion.div
            className="landing-footer-divider"
            initial={{ scaleX: 0, opacity: 0 }}
            whileInView={{ scaleX: 1, opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
          />
          <motion.div
            className="landing-footer-bottom"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.6, ease: [0.16, 1, 0.3, 1] }}
          >
            <p className="landing-footer-copy">&copy; 2026 AI Interview Coach. All rights reserved.</p>
            <div className="landing-footer-links">
              <a href="#">Project Docs</a>
              <a href="#">GitHub</a>
              <a href="#">Contact</a>
            </div>
          </motion.div>
        </div>
      </footer>
    </div>
  );
}
