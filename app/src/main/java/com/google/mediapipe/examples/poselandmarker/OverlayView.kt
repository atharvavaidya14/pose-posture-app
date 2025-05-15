/*
 * Copyright 2023 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.google.mediapipe.examples.poselandmarker

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class OverlayView(context: Context?, attrs: AttributeSet?) :
    View(context, attrs) {

    private var results: PoseLandmarkerResult? = null
    private var pointPaint = Paint()
    private var linePaint = Paint()

    private var scaleFactor: Float = 1f
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    var leftNeckAngle: Double = 0.0
    var rightNeckAngle: Double = 0.0

    init {
        initPaints()
    }

    fun clear() {
        results = null
        pointPaint.reset()
        linePaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        linePaint.color =
            ContextCompat.getColor(context!!, R.color.mp_color_primary)
        linePaint.strokeWidth = LANDMARK_STROKE_WIDTH
        linePaint.style = Paint.Style.STROKE

        pointPaint.color = Color.YELLOW
        pointPaint.strokeWidth = LANDMARK_STROKE_WIDTH
        pointPaint.style = Paint.Style.FILL
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)
        results?.let { poseLandmarkerResult ->
            for (landmark in poseLandmarkerResult.landmarks()) {
                calculateNeckAngles(landmark)
                for (normalizedLandmark in landmark) {
                    canvas.drawPoint(
                        normalizedLandmark.x() * imageWidth * scaleFactor,
                        normalizedLandmark.y() * imageHeight * scaleFactor,
                        pointPaint
                    )
                }

                val redLinePaint = Paint(linePaint).apply {
                    color = Color.RED
                }

                for (connection in PoseLandmarker.POSE_LANDMARKS) {
                    val start = landmark[connection!!.start()]
                    val end = landmark[connection.end()]

                    val startX = start.x() * imageWidth * scaleFactor
                    val startY = start.y() * imageHeight * scaleFactor
                    val endX = end.x() * imageWidth * scaleFactor
                    val endY = end.y() * imageHeight * scaleFactor

                    fun isElbowAboveShoulder(startIdx: Int, endIdx: Int, shoulderIdx: Int, elbowIdx: Int): Boolean {
                        return (startIdx == shoulderIdx && endIdx == elbowIdx || startIdx == elbowIdx && endIdx == shoulderIdx)
                                && landmark[elbowIdx].y() < landmark[shoulderIdx].y()
                    }
                    val useRed = isElbowAboveShoulder(connection.start(), connection.end(), 11, 13) ||
                            isElbowAboveShoulder(connection.start(), connection.end(), 12, 14)

                    canvas.drawLine(
                        startX,
                        startY,
                        endX,
                        endY,
                        if (useRed) redLinePaint else linePaint
                    )
                }
            }
        }
    }

    private fun calculateNeckAngles(landmarks: List<NormalizedLandmark>) {
        val leftShoulder = landmarks[11]
        val rightShoulder = landmarks[12]
        val leftEar = landmarks[7]
        val rightEar = landmarks[8]

        // Vertical reference for left shoulder
        val verticalLeftX = leftShoulder.x()
        val verticalLeftY = leftShoulder.y() - 0.1f

        // Vertical reference for right shoulder
        val verticalRightX = rightShoulder.x()
        val verticalRightY = rightShoulder.y() - 0.1f

        // Angle between left ear and left shoulder (left side neck tilt)
        leftNeckAngle = getAngle(
            leftEar.x(), leftEar.y(),
            leftShoulder.x(), leftShoulder.y(),
            verticalLeftX, verticalLeftY
        )

        // Angle between right ear and right shoulder (right side neck tilt)
        rightNeckAngle = getAngle(
            rightEar.x(), rightEar.y(),
            rightShoulder.x(), rightShoulder.y(),
            verticalRightX, verticalRightY
        )
    }

    // Util: Calculate angle between 3 points
    private fun getAngle(ax: Float, ay: Float, bx: Float, by: Float, cx: Float, cy: Float): Double {
        val ab = doubleArrayOf((ax - bx).toDouble(), (ay - by).toDouble())
        val cb = doubleArrayOf((cx - bx).toDouble(), (cy - by).toDouble())
        val dotProduct = ab[0] * cb[0] + ab[1] * cb[1]
        val abMag = sqrt(ab[0] * ab[0] + ab[1] * ab[1])
        val cbMag = sqrt(cb[0] * cb[0] + cb[1] * cb[1])
        return Math.toDegrees(acos(dotProduct / (abMag * cbMag)))
    }

    fun setResults(
        poseLandmarkerResults: PoseLandmarkerResult,
        imageHeight: Int,
        imageWidth: Int,
        runningMode: RunningMode = RunningMode.IMAGE
    ) {
        results = poseLandmarkerResults

        this.imageHeight = imageHeight
        this.imageWidth = imageWidth

        scaleFactor = when (runningMode) {
            RunningMode.IMAGE,
            RunningMode.VIDEO -> {
                min(width * 1f / imageWidth, height * 1f / imageHeight)
            }
            RunningMode.LIVE_STREAM -> {
                // PreviewView is in FILL_START mode. So we need to scale up the
                // landmarks to match with the size that the captured images will be
                // displayed.
                max(width * 1f / imageWidth, height * 1f / imageHeight)
            }
        }
        invalidate()
    }

    companion object {
        private const val LANDMARK_STROKE_WIDTH = 12F
    }
}