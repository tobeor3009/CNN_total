import SimpleITK as sitk


def get_fast_marching_filter_speed_image(input_array, sigma=0.5, alpha=-1, beta=1.0):
    # input_array: png array?
    inputImage = sitk.GetImageFromArray(input_array.astype("float32"))

    smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
    smoothing.SetTimeStep(0.125)
    smoothing.SetNumberOfIterations(5)
    smoothing.SetConductanceParameter(9.0)
    smoothingOutput = smoothing.Execute(inputImage)

    gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gradientMagnitude.SetSigma(sigma)
    gradientMagnitudeOutput = gradientMagnitude.Execute(smoothingOutput)

    sigmoid = sitk.SigmoidImageFilter()
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(alpha)
    sigmoid.SetBeta(beta)
    sigmoidOutput = sigmoid.Execute(gradientMagnitudeOutput)

    return 1 - sitk.GetArrayFromImage(sigmoidOutput)


def get_fast_marching_filter_result(input_array, seedPosition=(256, 256), sigma=0.5, alpha=-1, beta=1.0,
                                    timeThreshold=100, stoppingTime=110):
    # input_array: png array?
    inputImage = sitk.GetImageFromArray(input_array.astype("float32"))

    smoothing = sitk.CurvatureAnisotropicDiffusionImageFilter()
    smoothing.SetTimeStep(0.125)
    smoothing.SetNumberOfIterations(5)
    smoothing.SetConductanceParameter(9.0)
    smoothingOutput = smoothing.Execute(inputImage)

    gradientMagnitude = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    gradientMagnitude.SetSigma(sigma)
    gradientMagnitudeOutput = gradientMagnitude.Execute(smoothingOutput)

    sigmoid = sitk.SigmoidImageFilter()
    sigmoid.SetOutputMinimum(0.0)
    sigmoid.SetOutputMaximum(1.0)
    sigmoid.SetAlpha(alpha)
    sigmoid.SetBeta(beta)
    sigmoidOutput = sigmoid.Execute(gradientMagnitudeOutput)

    fastMarching = sitk.FastMarchingImageFilter()

    seedValue = 0
    trialPoint = (seedPosition[0], seedPosition[1], seedValue)

    fastMarching.AddTrialPoint(trialPoint)

    fastMarching.SetStoppingValue(stoppingTime)

    fastMarchingOutput = fastMarching.Execute(sigmoidOutput)

    thresholder = sitk.BinaryThresholdImageFilter()
    thresholder.SetLowerThreshold(0.0)
    thresholder.SetUpperThreshold(timeThreshold)
    thresholder.SetOutsideValue(0)
    thresholder.SetInsideValue(255)

    result = thresholder.Execute(fastMarchingOutput)

    image_dict = {"InputImage": inputImage,
                  "SpeedImage": sigmoidOutput,
                  "TimeCrossingMap": fastMarchingOutput,
                  "Segmentation": result,
                  }
