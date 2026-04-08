import React from 'react'
import { SignUp } from '@clerk/nextjs'

function SignUpPage() {
  return (
    <div className= "flex justify-center items-center min-h-screen bg-gray-50">
      <SignUp />
    </div>
  )
}

export default SignUpPage;