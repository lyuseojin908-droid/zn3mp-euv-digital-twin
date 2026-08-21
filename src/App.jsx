import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  Atom,
  Beaker,
  BrainCircuit,
  ChevronRight,
  CircleGauge,
  Database,
  FlaskConical,
  Gauge,
  Layers3,
  LineChart,
  Microscope,
  Orbit,
  Radar,
  ShieldCheck,
  Sparkles,
  Target,
  Thermometer,
  Waves,
} from 'lucide-react'

function cholesky(A) {
  const n = A.length
  const L = Array.from({ length: n }, () => Array(n).fill(0))

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0
      for (let k = 0; k < j; k++) sum += L[i][k] * L[j][k]

      if (i === j) {
        L[i][j] = Math.sqrt(Math.max(A[i][i] - sum, 1e-12))
      } else {
        L[i][j] = (A[i][j] - sum) / L[j][j]
      }
    }
  }
  return L
}

function forwardSolve(L, b) {
  const y = Array(b.length).fill(0)
  for (let i = 0; i < b.length; i++) {
    let sum = 0
    for (let j = 0; j < i; j++) sum += L[i][j] * y[j]
    y[i] = (b[i] - sum) / L[i][i]
  }
  return y
}

function backwardSolve(L, y) {
  const n = y.length
  const x = Array(n).fill(0)
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0
    for (let j = i + 1; j < n; j++) sum += L[j][i] * x[j]
    x[i] = (y[i] - sum) / L[i][i]
  }
  return x
}

function dot(a, b) {
  return a.reduce((sum, value, i) => sum + value * b[i], 0)
}

function buildPredictor(modelJson) {
  const trainingData = modelJson.training_data
  const xMean = modelJson.scaler.mean
  const xScale = modelJson.scaler.scale
  const yMean = modelJson.target_normalization.mean
  const yStd = modelJson.target_normalization.std
  const signal = modelJson.kernel.constant_value
  const lengthScale = modelJson.kernel.rbf_length_scale
  const noise = modelJson.kernel.noise_level

  const standardize = x => x.map((value, i) => (value - xMean[i]) / xScale[i])

  const kernel = (a, b) => {
    let q = 0
    for (let i = 0; i < a.length; i++) {
      const d = (a[i] - b[i]) / lengthScale[i]
      q += d * d
    }
    return signal * Math.exp(-0.5 * q)
  }

  const trainX = trainingData.map(d =>
    standardize([d.temperature_C, d.pressure_Torr, d.underlayer])
  )
  const trainY = trainingData.map(d =>
    (d.dev_rate_nm_cycle - yMean) / yStd
  )

  const K = trainX.map((a, i) =>
    trainX.map((b, j) => kernel(a, b) + (i === j ? noise : 0))
  )
  const L = cholesky(K)
  const alpha = backwardSolve(L, forwardSolve(L, trainY))

  return (temperature, pressure, underlayer) => {
    const x = standardize([temperature, pressure, underlayer])
    const kStar = trainX.map(xi => kernel(x, xi))
    const meanNorm = dot(kStar, alpha)
    const v = forwardSolve(L, kStar)
    const varNorm = Math.max(signal + noise - dot(v, v), 0)

    return {
      mean: yMean + yStd * meanNorm,
      std: yStd * Math.sqrt(varNorm),
    }
  }
}

function buildGrid(predictor, u, metric = 'mean', rows = 25, cols = 41) {
  const cells = []
  let min = Infinity
  let max = -Infinity

  for (let r = 0; r < rows; r++) {
    const p = 2.0 - (r / (rows - 1)) * 1.5
    for (let c = 0; c < cols; c++) {
      const t = 80 + (c / (cols - 1)) * 40
      const pred = predictor(t, p, u)
      const value = metric === 'mean' ? pred.mean : pred.std
      min = Math.min(min, value)
      max = Math.max(max, value)
      cells.push({ r, c, t, p, value })
    }
  }
  return { cells, min, max, rows, cols }
}

function colorScale(v, min, max, uncertainty = false) {
  const n = Math.max(0, Math.min(1, (v - min) / Math.max(max - min, 1e-9)))
  if (uncertainty) {
    const hue = 220 - n * 175
    return `hsl(${hue} 82% ${20 + n * 37}%)`
  }
  const hue = 220 - n * 170
  return `hsl(${hue} 84% ${20 + n * 40}%)`
}

function Heatmap({ predictor, trainingData, u, metric, t, p }) {
  const grid = useMemo(() => buildGrid(predictor, u, metric), [predictor, u, metric])
  const [hover, setHover] = useState(null)
  const x = ((t - 80) / 40) * 100
  const y = ((2 - p) / 1.5) * 100
  const experimental = trainingData.filter(d => d.underlayer === u)

  return (
    <div className="heatmap-shell">
      <div className="plot-title-row">
        <div>
          <div className="panel-eyebrow">{metric === 'mean' ? 'SURROGATE FIELD' : 'MODEL CONFIDENCE'}</div>
          <h3>{metric === 'mean' ? 'Predicted Development Rate' : 'Prediction Uncertainty'}</h3>
        </div>
        <div className="legend">
          <span><i className="dot exp-dot" /> experiment</span>
          <span><i className="dot op-dot" /> operating point</span>
        </div>
      </div>

      <div className="heatmap-area">
        <div className="y-label">hfacH Pressure (Torr)</div>
        <div className="heatmap-frame">
          <div
            className="heatmap-grid"
            style={{
              gridTemplateColumns: `repeat(${grid.cols}, 1fr)`,
              gridTemplateRows: `repeat(${grid.rows}, 1fr)`,
            }}
          >
            {grid.cells.map((cell, idx) => (
              <div
                key={idx}
                className="heat-cell"
                style={{
                  background: colorScale(
                    cell.value,
                    grid.min,
                    grid.max,
                    metric === 'std'
                  ),
                }}
                onMouseEnter={() => setHover(cell)}
                onMouseLeave={() => setHover(null)}
              />
            ))}
          </div>

          {experimental.map((d, i) => (
            <div
              key={i}
              className="exp-marker"
              style={{
                left: `${((d.temperature_C - 80) / 40) * 100}%`,
                top: `${((2 - d.pressure_Torr) / 1.5) * 100}%`,
              }}
              title={`${d.temperature_C} °C · ${d.pressure_Torr} Torr · ${d.dev_rate_nm_cycle} nm/cycle`}
            >
              ×
            </div>
          ))}

          <div className="op-marker" style={{ left: `${x}%`, top: `${y}%` }}>
            <span />
          </div>

          {hover && (
            <div
              className="hover-card"
              style={{
                left: `${Math.min(78, (hover.c / (grid.cols - 1)) * 100 + 2)}%`,
                top: `${Math.min(76, (hover.r / (grid.rows - 1)) * 100 + 2)}%`,
              }}
            >
              <b>{hover.t.toFixed(1)} °C</b>
              <span>{hover.p.toFixed(3)} Torr</span>
              <strong>
                {metric === 'mean'
                  ? `${hover.value.toFixed(2)} nm/cycle`
                  : `σ ${hover.value.toFixed(2)}`}
              </strong>
            </div>
          )}
        </div>
        <div className="x-label">Temperature (°C)</div>
      </div>

      <div className="scale-row">
        <span>{grid.min.toFixed(2)}</span>
        <div className={`gradient-bar ${metric === 'std' ? 'uncertainty' : ''}`} />
        <span>{grid.max.toFixed(2)}</span>
      </div>
    </div>
  )
}

function MetricCard({ icon: Icon, label, value, unit, accent = 'cyan', foot }) {
  return (
    <div className="metric-card">
      <div className={`metric-icon ${accent}`}><Icon size={18} /></div>
      <div className="metric-label">{label}</div>
      <div className="metric-main">
        <span>{value}</span>
        {unit && <small>{unit}</small>}
      </div>
      <div className="metric-foot">{foot}</div>
    </div>
  )
}

function ProcessSchematic({ u }) {
  const steps = [
    { icon: Layers3, title: 'MLD Resist', sub: 'Zn–3MP' },
    { icon: Atom, title: 'EUV Exposure', sub: '13.5 nm' },
    { icon: Waves, title: 'Dry Develop', sub: 'hfacH' },
    { icon: ShieldCheck, title: 'Pattern', sub: u ? 'Al₂O₃ integrated' : 'No underlayer' },
  ]

  return (
    <div className="process-schematic">
      {steps.map((s, i) => (
        <div className="process-step-wrap" key={s.title}>
          <div className="process-step">
            <div className="step-icon"><s.icon size={18} /></div>
            <div>
              <b>{s.title}</b>
              <span>{s.sub}</span>
            </div>
          </div>
          {i < steps.length - 1 && <ChevronRight className="process-arrow" size={18} />}
        </div>
      ))}
    </div>
  )
}

function ExperimentPlanner({ model, predictor, u }) {
  const top = useMemo(() => {
    if (u === 1 && model.recommended_experiments?.length) {
      return model.recommended_experiments.slice(0, 6).map(r => ({
        t: r.temperature_C,
        p: r.pressure_Torr,
        mean: r.pred_rate,
        std: r.uncertainty,
        ucb: r.ucb_score,
      }))
    }

    const candidates = []
    for (let ti = 0; ti <= 80; ti++) {
      const t = 80 + ti * 0.5
      for (let pi = 0; pi <= 60; pi++) {
        const p = 0.5 + pi * 0.025
        const pred = predictor(t, p, u)
        const close = model.training_data
          .filter(d => d.underlayer === u)
          .some(d => Math.abs(t - d.temperature_C) < 1.0 && Math.abs(p - d.pressure_Torr) < 0.05)

        if (!close) {
          candidates.push({ t, p, mean: pred.mean, std: pred.std, ucb: pred.mean + pred.std })
        }
      }
    }
    return candidates.sort((a, b) => b.ucb - a.ucb).slice(0, 6)
  }, [model, predictor, u])

  return (
    <div className="planner-grid">
      <div className="planner-main">
        <div className="panel-eyebrow">ACTIVE LEARNING QUEUE</div>
        <h3>Next Experiment Candidates</h3>
        <p className="panel-copy">
          UCB = predicted rate + uncertainty. High-ranking points combine performance potential with information value.
        </p>

        <div className="experiment-list">
          {top.map((r, i) => (
            <div className="experiment-row" key={i}>
              <div className="rank">0{i + 1}</div>
              <div>
                <b>{r.t.toFixed(1)} °C</b>
                <span>{r.p.toFixed(3)} Torr</span>
              </div>
              <div>
                <b>{r.mean.toFixed(2)}</b>
                <span>nm/cycle</span>
              </div>
              <div>
                <b>± {r.std.toFixed(2)}</b>
                <span>uncertainty</span>
              </div>
              <div className="ucb-chip">{r.ucb.toFixed(2)} UCB</div>
            </div>
          ))}
        </div>
      </div>

      <div className="planner-side">
        <div className="mini-panel accent-panel">
          <Target size={20} />
          <span>Exploration</span>
          <b>Highest UCB candidate</b>
          <p>Searches for high predicted performance and high information value.</p>
        </div>
        <div className="mini-panel">
          <Radar size={20} />
          <span>Validation</span>
          <b>Uncertainty-aware</b>
          <p>Uses posterior σ instead of treating every prediction as equally reliable.</p>
        </div>
        <div className="mini-panel">
          <Microscope size={20} />
          <span>Colab linked</span>
          <b>JSON model source</b>
          <p>The dashboard loads scaler, kernel, metrics and experiment ranks from Colab output.</p>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const [model, setModel] = useState(null)
  const [dft, setDft] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [dftError, setDftError] = useState('')
  const [temperature, setTemperature] = useState(95)
  const [pressure, setPressure] = useState(0.95)
  const [underlayer, setUnderlayer] = useState(1)
  const [view, setView] = useState('twin')

  useEffect(() => {
    fetch('/digital_twin_model.json')
      .then(response => {
        if (!response.ok) throw new Error('digital_twin_model.json load failed')
        return response.json()
      })
      .then(setModel)
      .catch(error => setLoadError(error.message))

    fetch('/dft_descriptors.json')
      .then(response => {
        if (!response.ok) throw new Error('dft_descriptors.json load failed')
        return response.json()
      })
      .then(setDft)
      .catch(error => setDftError(error.message))
  }, [])

  const predictor = useMemo(() => model ? buildPredictor(model) : null, [model])
  const pred = useMemo(
    () => predictor ? predictor(temperature, pressure, underlayer) : null,
    [predictor, temperature, pressure, underlayer]
  )

  if (loadError) {
    return <div style={{padding:'40px',color:'#fff'}}>Model load error: {loadError}</div>
  }
  if (!model || !predictor || !pred) {
    return <div style={{padding:'40px',color:'#fff'}}>Loading Colab AI model…</div>
  }

  const GPR_MAE = model.model.metrics.gpr_mae
  const LINEAR_MAE = model.model.metrics.linear_mae
  const confidence = pred.std < 0.25 ? 'HIGH' : pred.std < 0.7 ? 'MEDIUM' : 'LOW'
  const improvement = Math.round((1 - GPR_MAE / LINEAR_MAE) * 100)
  const dftReady = Boolean(dft?.reaction_summary?.length && dft?.precursor_descriptor_summary?.length)
  const dftFunctionals = dftReady
    ? [...new Set((dft.reaction_functional_results || []).map(r => r.functional))]
    : []
  const r1 = dft?.reaction_summary?.find(r => r.reaction_id === 'R1')
  const r2 = dft?.reaction_summary?.find(r => r.reaction_id === 'R2')
  const dez = dft?.precursor_descriptor_summary?.find(r => r.species === 'DEZ')
  const mp3 = dft?.precursor_descriptor_summary?.find(r => r.species === '3MP')

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-orbit"><Orbit size={22} /></div>
          <div>
            <span>AI PROCESS INTELLIGENCE</span>
            <b>Zn–3MP Digital Twin</b>
          </div>
        </div>

        <div className="side-section">
          <div className="side-head">
            <CircleGauge size={16} />
            <span>Operating Point</span>
          </div>

          <label>
            <div className="label-row"><span>Temperature</span><strong>{temperature.toFixed(1)} °C</strong></div>
            <input type="range" min="80" max="120" step="0.5" value={temperature}
              onChange={e => setTemperature(Number(e.target.value))} />
          </label>

          <label>
            <div className="label-row"><span>hfacH Pressure</span><strong>{pressure.toFixed(3)} Torr</strong></div>
            <input type="range" min="0.5" max="2" step="0.025" value={pressure}
              onChange={e => setPressure(Number(e.target.value))} />
          </label>

          <div className="toggle-box">
            <div><span>Al₂O₃ Interface</span><small>{underlayer ? 'Integrated' : 'None'}</small></div>
            <button aria-label="Toggle Al2O3 underlayer" className={underlayer ? 'switch on' : 'switch'}
              onClick={() => setUnderlayer(v => (v ? 0 : 1))}><i /></button>
          </div>
        </div>

        <div className="side-section model-block">
          <div className="side-head"><BrainCircuit size={16} /><span>AI Engine</span></div>
          <div className="model-line"><span>Algorithm</span><b>Gaussian Process</b></div>
          <div className="model-line"><span>Validation</span><b>LOOCV</b></div>
          <div className="model-line"><span>Inputs</span><b>{model.model.features.length} variables</b></div>
          <div className="model-line"><span>Training</span><b>{model.training_data.length} points</b></div>
        </div>

        <div className="sidebar-note">
          <Sparkles size={15} />
          <p>
            Colab-trained scaler, GPR kernel, validation metrics and active-learning candidates are loaded from JSON.
          </p>
        </div>
      </aside>

      <main className="main">
        <header className="hero">
          <div className="hero-left">
            <div className="hero-kicker"><span className="online-dot" />MODEL ONLINE<i />COLAB → REACT → NETLIFY</div>
            <h1>AI-Driven ALD/MLD<br/><em>Process Digital Twin</em></h1>
            <p>
              Gaussian Process Regression predicts dry-development behavior, quantifies posterior uncertainty,
              and ranks follow-up experiments. Real DFT ensemble descriptors are loaded as a physics-informed AI layer.
            </p>
          </div>

          <div className="hero-right">
            <div className="signal-card">
              <div className="signal-head"><span>LIVE AI SENSOR</span><Activity size={17} /></div>
              <div className="signal-value">{pred.mean.toFixed(2)}<small>nm/cycle</small></div>
              <div className="wave-line">
                {Array.from({ length: 22 }).map((_, i) => (
                  <i key={i} style={{ height: `${22 + Math.abs(Math.sin(i * 0.63)) * 34}px` }} />
                ))}
              </div>
              <div className="signal-footer"><span>σ {pred.std.toFixed(3)}</span><b>{confidence} CONFIDENCE</b></div>
            </div>
          </div>
        </header>

        <ProcessSchematic u={underlayer} />

        <section className="metrics">
          <MetricCard icon={Gauge} label="Predicted Rate" value={pred.mean.toFixed(2)} unit="nm/cycle"
            foot={`${temperature.toFixed(1)} °C · ${pressure.toFixed(3)} Torr`} />
          <MetricCard icon={Radar} label="Uncertainty" value={`± ${pred.std.toFixed(2)}`} accent="amber"
            foot="posterior standard deviation" />
          <MetricCard icon={ShieldCheck} label="Model Confidence" value={confidence}
            accent={confidence === 'HIGH' ? 'green' : 'amber'} foot="uncertainty-aware estimate" />
          <MetricCard icon={LineChart} label="GPR vs Linear MAE" value={`−${improvement}%`} accent="green"
            foot={`${GPR_MAE.toFixed(3)} vs ${LINEAR_MAE.toFixed(3)} nm/cycle`} />
        </section>

        <nav className="view-tabs">
          <button className={view === 'twin' ? 'active' : ''} onClick={() => setView('twin')}><Waves size={16} /> Process Twin</button>
          <button className={view === 'uncertainty' ? 'active' : ''} onClick={() => setView('uncertainty')}><Radar size={16} /> Uncertainty Field</button>
          <button className={view === 'planner' ? 'active' : ''} onClick={() => setView('planner')}><Target size={16} /> Active Learning</button>
          <button className={view === 'dft' ? 'active' : ''} onClick={() => setView('dft')}><Atom size={16} /> DFT → AI</button>
          <button className={view === 'data' ? 'active' : ''} onClick={() => setView('data')}><Database size={16} /> Model Data</button>
        </nav>

        <section className="workspace">
          {view === 'twin' && (
            <div className="workspace-grid">
              <Heatmap predictor={predictor} trainingData={model.training_data} u={underlayer} metric="mean" t={temperature} p={pressure} />
              <div className="insight-stack">
                <div className="insight-card">
                  <div className="panel-eyebrow">OPERATING STATE</div><h3>Live Point</h3>
                  <div className="state-grid">
                    <div><Thermometer size={16}/><span>Temperature</span><b>{temperature.toFixed(1)} °C</b></div>
                    <div><Beaker size={16}/><span>Pressure</span><b>{pressure.toFixed(3)} Torr</b></div>
                    <div><Layers3 size={16}/><span>Interface</span><b>{underlayer ? 'Al₂O₃' : 'None'}</b></div>
                    <div><Activity size={16}/><span>Rate</span><b>{pred.mean.toFixed(2)}</b></div>
                  </div>
                </div>

                <div className="insight-card">
                  <div className="panel-eyebrow">AI READOUT</div>
                  <h3>{pred.std < 0.25 ? 'Stable interpolation zone' : pred.std < 0.7 ? 'Moderate model risk' : 'Verification recommended'}</h3>
                  <p>{pred.std < 0.25
                    ? 'The selected point is supported by nearby experimental conditions.'
                    : pred.std < 0.7
                    ? 'The model can estimate this region, but uncertainty is no longer negligible.'
                    : 'The selected operating point lies in a sparse region of the experimental design space.'}</p>
                </div>

                <div className="insight-card luminous">
                  <FlaskConical size={20} /><div><span>NEXT BEST ACTION</span><b>Use active learning</b><p>Rank experiments with prediction + uncertainty.</p></div>
                </div>
              </div>
            </div>
          )}

          {view === 'uncertainty' && (
            <div className="workspace-grid">
              <Heatmap predictor={predictor} trainingData={model.training_data} u={underlayer} metric="std" t={temperature} p={pressure} />
              <div className="insight-stack">
                <div className="insight-card"><div className="panel-eyebrow">UNCERTAINTY LOGIC</div><h3>AI knows what it does not know</h3><p>High-σ regions become experiment candidates instead of being treated as equally reliable predictions.</p></div>
                <div className="insight-card"><div className="panel-eyebrow">CURRENT σ</div><div className="big-number">{pred.std.toFixed(3)}</div><p>Posterior standard deviation at the selected operating point.</p></div>
              </div>
            </div>
          )}

          {view === 'planner' && <ExperimentPlanner model={model} predictor={predictor} u={underlayer} />}

          {view === 'dft' && (
            <div className="data-panel">
              <div className="data-head">
                <div>
                  <div className="panel-eyebrow">REAL QUANTUM DESCRIPTOR LAYER</div>
                  <h3>DFT → AI Physics Features</h3>
                </div>
                <div className="dataset-chip">{dftReady ? 'REAL DFT LOADED' : 'DFT FILE NOT LOADED'}</div>
              </div>

              {!dftReady ? (
                <div className="insight-card" style={{marginTop:'14px'}}>
                  <div className="panel-eyebrow">LOAD STATUS</div>
                  <h3>Waiting for dft_descriptors.json</h3>
                  <p>{dftError || 'Place dft_descriptors.json in the public folder and redeploy.'}</p>
                </div>
              ) : (
                <>
                  <div className="validation-strip">
                    <div><span>DFT ENSEMBLE</span><b>{dftFunctionals.length} functionals</b></div>
                    <div><span>R1 ΔE</span><b>{r1?.reaction_energy_mean_eV.toFixed(3)} eV</b></div>
                    <div><span>R1 σDFT</span><b>{r1?.reaction_energy_std_eV.toFixed(3)} eV</b></div>
                    <div><span>R2 ΔE</span><b>{r2?.reaction_energy_mean_eV.toFixed(3)} eV</b></div>
                    <div><span>R2 σDFT</span><b>{r2?.reaction_energy_std_eV.toFixed(3)} eV</b></div>
                  </div>

                  <div className="workspace-grid" style={{marginTop:'14px'}}>
                    <div className="insight-card">
                      <div className="panel-eyebrow">PRECURSOR DESCRIPTORS</div>
                      <h3>Electronic Structure Ensemble</h3>
                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr><th>Species</th><th>HOMO</th><th>LUMO</th><th>Gap</th><th>Dipole</th></tr>
                          </thead>
                          <tbody>
                            {[dez, mp3].filter(Boolean).map(x => (
                              <tr key={x.species}>
                                <td>{x.species}</td>
                                <td>{x.homo_mean_eV.toFixed(3)} eV</td>
                                <td>{x.lumo_mean_eV.toFixed(3)} eV</td>
                                <td>{x.gap_mean_eV.toFixed(3)} eV</td>
                                <td>{x.dipole_mean_D.toFixed(3)} D</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <div className="insight-stack">
                      <div className="insight-card">
                        <div className="panel-eyebrow">R1 · DEZ LIGAND EXCHANGE</div>
                        <h3>{r1?.reaction_energy_mean_eV.toFixed(3)} ± {r1?.reaction_energy_std_eV.toFixed(3)} eV</h3>
                        <p>Mean reaction energy ± functional ensemble spread. The spread is treated as a DFT uncertainty descriptor.</p>
                      </div>

                      <div className="insight-card">
                        <div className="panel-eyebrow">R2 · 3MP LIGAND EXCHANGE</div>
                        <h3>{r2?.reaction_energy_mean_eV.toFixed(3)} ± {r2?.reaction_energy_std_eV.toFixed(3)} eV</h3>
                        <p>R2 shows the larger functional sensitivity, so the quantum calculation itself carries more model-form uncertainty.</p>
                      </div>

                      <div className="insight-card luminous">
                        <BrainCircuit size={20} />
                        <div>
                          <span>AI INTEGRATION STATUS</span>
                          <b>Physics layer ready</b>
                          <p>Next: calculate an activation barrier so the DFT information can vary with temperature instead of being repeated across all 12 process rows.</p>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="insight-card" style={{marginTop:'14px'}}>
                    <div className="panel-eyebrow">SCIENTIFIC SCOPE</div>
                    <h3>Molecular surrogate, not periodic surface DFT</h3>
                    <p>{dft.model_scope?.warning}</p>
                    <p style={{marginTop:'8px'}}>
                      Surface proxy: {dft.model_scope?.surface_proxy}. Activation barrier: {dft.activation_barrier?.status?.replaceAll('_',' ')}.
                    </p>
                  </div>

                  <div className="insight-card" style={{marginTop:'14px'}}>
                    <div className="panel-eyebrow">NEXT AI EXPERIMENT</div>
                    <h3>Process-only GPR → Physics-informed GPR</h3>
                    <p>
                      The present 12 process points use the same Zn–3MP chemistry, so static DFT values cannot simply be copied into every row.
                      The next meaningful feature is a kinetics-aware descriptor derived from the activation barrier and temperature, followed by the same LOOCV comparison.
                    </p>
                  </div>
                </>
              )}
            </div>
          )}

          {view === 'data' && (
            <div className="data-panel">
              <div className="data-head">
                <div><div className="panel-eyebrow">TRAINING FOUNDATION</div><h3>Process Dataset</h3></div>
                <div className="dataset-chip">{model.training_data.length} POINTS</div>
              </div>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Temperature</th><th>Pressure</th><th>Al₂O₃</th><th>Development rate</th><th>Source</th></tr></thead>
                  <tbody>
                    {model.training_data.map((d, i) => (
                      <tr key={i}>
                        <td>{d.temperature_C.toFixed(1)} °C</td>
                        <td>{d.pressure_Torr.toFixed(3)} Torr</td>
                        <td>{d.underlayer ? 'Integrated' : 'None'}</td>
                        <td>{d.dev_rate_nm_cycle.toFixed(2)} nm/cycle</td>
                        <td><span className={`source-chip ${d.source}`}>{d.source}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="validation-strip">
                <div><span>GPR LOOCV MAE</span><b>{GPR_MAE.toFixed(3)}</b></div>
                <div><span>Linear LOOCV MAE</span><b>{LINEAR_MAE.toFixed(3)}</b></div>
                <div><span>Error reduction</span><b>{improvement}%</b></div>
              </div>
            </div>
          )}
        </section>

        <footer>
          <span>AI-Driven Zn–3MP Process Digital Twin</span>
          <span>Colab-trained GPR · uncertainty-aware prediction · active learning · real DFT descriptor layer</span>
        </footer>
      </main>
    </div>
  )
}
