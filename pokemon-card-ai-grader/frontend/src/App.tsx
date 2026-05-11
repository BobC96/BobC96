import { useState } from 'react'

type GradeResponse = {
  estimated_grade: number
  combined_score: number
  confidence: number
  recommendation: string
  disclaimer: string
  front: any
  back: any
}

export default function App() {
  const [frontImage, setFrontImage] = useState<File | null>(null)
  const [backImage, setBackImage] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<GradeResponse | null>(null)
  const [error, setError] = useState('')

  const submitGrade = async () => {
    if (!frontImage || !backImage) {
      setError('Please upload both front and back scans.')
      return
    }

    setLoading(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('front_image', frontImage)
      formData.append('back_image', backImage)

      const response = await fetch('http://localhost:8000/grade', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Failed to process grading request')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError('Unable to connect to backend API.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <h1>Pokémon Card AI Grader</h1>

      <div className="upload-section">
        <div className="upload-box">
          <h2>Front Scan</h2>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setFrontImage(e.target.files?.[0] || null)}
          />
        </div>

        <div className="upload-box">
          <h2>Back Scan</h2>
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setBackImage(e.target.files?.[0] || null)}
          />
        </div>
      </div>

      <button onClick={submitGrade} disabled={loading}>
        {loading ? 'Analyzing...' : 'Grade Card'}
      </button>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result-card">
          <h2>Estimated PSA Grade: {result.estimated_grade}</h2>

          <div className="score-grid">
            <div>
              <strong>Combined Score</strong>
              <p>{result.combined_score}</p>
            </div>

            <div>
              <strong>Confidence</strong>
              <p>{Math.round(result.confidence * 100)}%</p>
            </div>

            <div>
              <strong>Recommendation</strong>
              <p>{result.recommendation}</p>
            </div>
          </div>

          <div className="details-grid">
            <div>
              <h3>Front Scores</h3>
              <p>Centering: {result.front.scores.centering}</p>
              <p>Corners: {result.front.scores.corners}</p>
              <p>Edges: {result.front.scores.edges}</p>
              <p>Surface: {result.front.scores.surface}</p>
            </div>

            <div>
              <h3>Back Scores</h3>
              <p>Centering: {result.back.scores.centering}</p>
              <p>Corners: {result.back.scores.corners}</p>
              <p>Edges: {result.back.scores.edges}</p>
              <p>Surface: {result.back.scores.surface}</p>
            </div>
          </div>

          <p className="disclaimer">{result.disclaimer}</p>
        </div>
      )}
    </div>
  )
}
